#!/usr/bin/env bash
# ── auto_stop.sh ──────────────────────────────────────────────────────────────
# Kill the live scanner, backfill EOD data, replay the day, parse results,
# and send a formatted Telegram daily summary.
# Called by launchd at 1:05 PM PT (Mon–Fri) or manually.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── run_with_timeout: portable timeout for macOS (no coreutils needed) ───────
# Usage: run_with_timeout <seconds> <command> [args...]
# Returns: 0 on success, 124 on timeout, or the command's exit code.
run_with_timeout() {
    local timeout_secs="$1"; shift
    "$@" &
    local cmd_pid=$!
    ( sleep "$timeout_secs" && kill "$cmd_pid" 2>/dev/null ) &
    local watcher_pid=$!
    if wait "$cmd_pid" 2>/dev/null; then
        kill "$watcher_pid" 2>/dev/null || true
        wait "$watcher_pid" 2>/dev/null || true
        return 0
    else
        local exit_code=$?
        kill "$watcher_pid" 2>/dev/null || true
        wait "$watcher_pid" 2>/dev/null || true
        # 143 = killed by SIGTERM (from our watcher) → treat as timeout
        if [[ $exit_code -eq 143 ]]; then
            return 124
        fi
        return $exit_code
    fi
}

PROJECT_DIR="$HOME/Desktop/I/Projects/spy-signal-platform"
VENV_DIR="$PROJECT_DIR/.venv"
LOGS_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOGS_DIR/scanner.pid"
TODAY=$(date +%Y-%m-%d)
LOG_FILE="$LOGS_DIR/stop_${TODAY}.log"

# ── Ensure logs directory exists ──────────────────────────────────────────────
mkdir -p "$LOGS_DIR"

exec >> "$LOG_FILE" 2>&1
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] auto_stop.sh BEGIN"
echo "============================================================"

# ── Load Telegram credentials from .env ───────────────────────────────────────
if [[ -f "$PROJECT_DIR/.env" ]]; then
    TELEGRAM_BOT_TOKEN=$(grep -E '^TELEGRAM_BOT_TOKEN=' "$PROJECT_DIR/.env" | cut -d= -f2-)
    TELEGRAM_CHAT_ID=$(grep -E '^TELEGRAM_CHAT_ID=' "$PROJECT_DIR/.env" | cut -d= -f2-)
else
    echo "WARNING: .env not found — Telegram summary will be skipped."
    TELEGRAM_BOT_TOKEN=""
    TELEGRAM_CHAT_ID=""
fi

# ── Kill the scanner ─────────────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    SCANNER_PID=$(cat "$PID_FILE")
    if kill -0 "$SCANNER_PID" 2>/dev/null; then
        echo "Stopping scanner (PID $SCANNER_PID)..."
        kill "$SCANNER_PID"
        # Wait up to 10 seconds for graceful shutdown
        for i in $(seq 1 10); do
            if ! kill -0 "$SCANNER_PID" 2>/dev/null; then
                echo "Scanner stopped after ${i}s."
                break
            fi
            sleep 1
        done
        # Force kill if still running
        if kill -0 "$SCANNER_PID" 2>/dev/null; then
            echo "Force killing scanner (PID $SCANNER_PID)..."
            kill -9 "$SCANNER_PID" 2>/dev/null || true
        fi
    else
        echo "Scanner not running (PID $SCANNER_PID already exited)."
    fi
    rm -f "$PID_FILE"
else
    echo "No PID file found — scanner was not running."
fi

# ── Activate virtual environment ──────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"

# ── Backfill EOD data (all tickers from config) ─────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Running EOD data backfill (last 2 days, all tickers)..."
ALL_SYMBOLS=$(python -c "from src.config import get_app_settings; print(','.join(get_app_settings().symbols))" 2>/dev/null) || ALL_SYMBOLS="SPY"
echo "  Tickers: $ALL_SYMBOLS"
if run_with_timeout 120 python scripts/backfill_data.py --days 2 --symbols "$ALL_SYMBOLS"; then
    echo "[$(date '+%H:%M:%S')] Backfill complete."
else
    BACKFILL_EXIT=$?
    if [[ $BACKFILL_EXIT -eq 124 ]]; then
        echo "WARNING: EOD backfill timed out after 120s."
    else
        echo "WARNING: EOD backfill failed (exit $BACKFILL_EXIT)."
    fi
fi

# ── Replay today's trading day ────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Running replay (this takes 2-3 minutes)..."
REPLAY_TIMED_OUT=false
REPLAY_TMPFILE=$(mktemp)
trap 'rm -f "$REPLAY_TMPFILE"' EXIT
if run_with_timeout 300 python scripts/replay_day.py --date "$TODAY" > "$REPLAY_TMPFILE" 2>&1; then
    REPLAY_OUTPUT=$(cat "$REPLAY_TMPFILE")
    echo "$REPLAY_OUTPUT"
else
    REPLAY_EXIT=$?
    REPLAY_OUTPUT=$(cat "$REPLAY_TMPFILE" 2>/dev/null || echo "")
    if [[ $REPLAY_EXIT -eq 124 ]]; then
        echo "WARNING: Replay timed out after 300s."
        REPLAY_TIMED_OUT=true
        REPLAY_OUTPUT=""
    else
        echo "WARNING: Replay failed (exit $REPLAY_EXIT)."
        echo "$REPLAY_OUTPUT"
    fi
fi

# ── Parse replay output for Telegram summary ─────────────────────────────────
if [[ "$REPLAY_TIMED_OUT" == "true" ]]; then
    # Replay timed out — send a minimal summary
    SIGNALS="?"
    WINS="?"
    LOSSES="?"
    WIN_RATE="?"
    NET_PNL="?"
    DAY_EMOJI="⚠️"
    TICKER_LINES=""
else
    # Extract summary stats from the replay output
    SIGNALS=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Signals:' | awk '{print $NF}' || echo "0")
    WINS=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Wins:' | awk '{print $NF}' || echo "0")
    LOSSES=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Losses:' | awk '{print $NF}' || echo "0")
    WIN_RATE=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Win Rate:' | awk '{print $NF}' || echo "0%")
    NET_PNL=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Net P&L:' | awk '{print $NF}' || echo "$0.00")

    # Determine day emoji
    if [[ "$NET_PNL" == *"-"* ]]; then
        DAY_EMOJI="🔴"
    elif [[ "$NET_PNL" == "$0.00" ]] || [[ "$SIGNALS" == "0" ]]; then
        DAY_EMOJI="⚪"
    else
        DAY_EMOJI="🟢"
    fi

    # Extract per-ticker lines (the numbered trade rows from replay output)
    TICKER_LINES=""
    while IFS= read -r line; do
        # Match lines like:  1  09:36  SPY   LONG  ...
        if echo "$line" | grep -qE '^\s+[0-9]+\s+[0-9]{2}:[0-9]{2}'; then
            TICKER=$(echo "$line" | awk '{print $3}')
            DIR=$(echo "$line" | awk '{print $4}')
            PNL=$(echo "$line" | awk '{print $(NF-1)}')
            OUTCOME=$(echo "$line" | awk '{print $8}')

            if [[ "$PNL" == *"-"* ]]; then
                T_EMOJI="❌"
            else
                T_EMOJI="✅"
            fi
            TICKER_LINES="${TICKER_LINES}${T_EMOJI} ${TICKER} ${DIR} → ${OUTCOME} (${PNL})"$'\n'
        fi
    done <<< "$REPLAY_OUTPUT"
fi

# ── Build Telegram message ────────────────────────────────────────────────────
DAY_NAME=$(date -j -f "%Y-%m-%d" "$TODAY" "+%A" 2>/dev/null || date -d "$TODAY" "+%A" 2>/dev/null || echo "")

if [[ "$REPLAY_TIMED_OUT" == "true" ]]; then
    MESSAGE="⚠️ *ORB Daily Summary — ${DAY_NAME} ${TODAY}*

⏱ Replay timed out after 5 minutes.
Scanner ran today but results could not be computed.
Run manually: \`python scripts/replay_day.py --date ${TODAY}\`"
else
    MESSAGE="${DAY_EMOJI} *ORB Daily Summary — ${DAY_NAME} ${TODAY}*

📊 *Stats*
• Signals: ${SIGNALS}
• Wins: ${WINS} | Losses: ${LOSSES}
• Win Rate: ${WIN_RATE}
• Net P&L: ${NET_PNL}

📋 *Trades*
${TICKER_LINES:-No signals fired today.}"
fi

echo ""
echo "──── Telegram Message ────"
echo "$MESSAGE"
echo "──────────────────────────"

# ── Send Telegram summary ─────────────────────────────────────────────────────
if [[ -n "$TELEGRAM_BOT_TOKEN" ]] && [[ -n "$TELEGRAM_CHAT_ID" ]]; then
    echo ""
    echo "[$(date '+%H:%M:%S')] Sending Telegram daily summary..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="$TELEGRAM_CHAT_ID" \
        -d parse_mode="Markdown" \
        --data-urlencode text="$MESSAGE")

    if [[ "$HTTP_CODE" == "200" ]]; then
        echo "Telegram summary sent successfully."
    else
        echo "WARNING: Telegram send failed (HTTP $HTTP_CODE)."
    fi
else
    echo "Telegram credentials not set — skipping summary."
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] auto_stop.sh DONE"
echo "============================================================"
