#!/usr/bin/env bash
# ── auto_stop.sh ──────────────────────────────────────────────────────────────
# Kill the live scanner, backfill EOD data, query live signals, replay the day,
# and send a two-section Telegram summary separating LIVE from BACKTEST results.
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
DB_PATH="$PROJECT_DIR/data/spy_signals.db"
# Use ET for all date calculations so "trading day" matches market hours
# regardless of the operator's local timezone (e.g. Pacific).
TODAY=$(TZ=America/New_York date +%Y-%m-%d)
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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LIVE PAPER TRADES — what actually happened today
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[$(date '+%H:%M:%S')] Querying live signals from database..."

LIVE_SECTION=""
LIVE_SIGNAL_COUNT=0
LIVE_APPROVED_COUNT=0
LIVE_TRADE_LINES=""

if [[ -f "$DB_PATH" ]]; then
    # Count all signals generated today (approved + rejected)
    LIVE_SIGNAL_COUNT=$(sqlite3 "$DB_PATH" \
        "SELECT COUNT(*) FROM signals WHERE timestamp >= '${TODAY}T00:00:00';" 2>/dev/null || echo "0")

    # Count approved signals (trades the risk manager allowed)
    LIVE_APPROVED_COUNT=$(sqlite3 "$DB_PATH" \
        "SELECT COUNT(*) FROM signals WHERE timestamp >= '${TODAY}T00:00:00' AND approved = 1;" 2>/dev/null || echo "0")

    # Get rejected reasons summary
    LIVE_REJECTED=$(sqlite3 "$DB_PATH" \
        "SELECT reject_reason, COUNT(*) as cnt FROM signals
         WHERE timestamp >= '${TODAY}T00:00:00' AND approved = 0
         GROUP BY reject_reason ORDER BY cnt DESC LIMIT 5;" 2>/dev/null || echo "")

    # Get approved trade details
    LIVE_TRADE_LINES=$(sqlite3 -separator '|' "$DB_PATH" \
        "SELECT direction, entry_price, stop_price, target_price, confidence, reason
         FROM signals
         WHERE timestamp >= '${TODAY}T00:00:00' AND approved = 1
         ORDER BY timestamp;" 2>/dev/null || echo "")

    # Get trades table entries (manually logged P&L from executor)
    LIVE_PNL_LINES=$(sqlite3 -separator '|' "$DB_PATH" \
        "SELECT direction, entry_price, target_price, pnl
         FROM trades
         WHERE timestamp >= '${TODAY}T00:00:00'
         ORDER BY timestamp;" 2>/dev/null || echo "")

    echo "  Signals today: $LIVE_SIGNAL_COUNT (approved: $LIVE_APPROVED_COUNT)"
fi

# Build live section for Telegram
if [[ "$LIVE_APPROVED_COUNT" -gt 0 ]] && [[ -n "$LIVE_TRADE_LINES" ]]; then
    LIVE_DETAILS=""
    while IFS='|' read -r dir entry stop target conf reason; do
        if [[ "$dir" == "LONG" ]]; then
            LIVE_DETAILS="${LIVE_DETAILS}  🟢 ${dir} \$${entry} (stop \$${stop}, target \$${target}) ★${conf}"$'\n'
        else
            LIVE_DETAILS="${LIVE_DETAILS}  🔴 ${dir} \$${entry} (stop \$${stop}, target \$${target}) ★${conf}"$'\n'
        fi
    done <<< "$LIVE_TRADE_LINES"

    # Add realized P&L from trades table if available
    LIVE_REALIZED_PNL=""
    if [[ -n "$LIVE_PNL_LINES" ]]; then
        TOTAL_PNL=$(sqlite3 "$DB_PATH" \
            "SELECT COALESCE(SUM(CAST(pnl AS REAL)), 0) FROM trades WHERE timestamp >= '${TODAY}T00:00:00';" 2>/dev/null || echo "0")
        TRADE_COUNT=$(sqlite3 "$DB_PATH" \
            "SELECT COUNT(*) FROM trades WHERE timestamp >= '${TODAY}T00:00:00';" 2>/dev/null || echo "0")
        if [[ "$TOTAL_PNL" == *"-"* ]]; then
            LIVE_REALIZED_PNL="  Realized: \$${TOTAL_PNL} (${TRADE_COUNT} closed)"
        else
            LIVE_REALIZED_PNL="  Realized: +\$${TOTAL_PNL} (${TRADE_COUNT} closed)"
        fi
    fi

    LIVE_SECTION="✅ *LIVE PAPER TRADES*
${LIVE_APPROVED_COUNT} signal(s) approved by risk manager
${LIVE_DETAILS}${LIVE_REALIZED_PNL}"
else
    # No live trades — explain why
    REJECT_SUMMARY=""
    if [[ "$LIVE_SIGNAL_COUNT" -gt 0 ]] && [[ -n "$LIVE_REJECTED" ]]; then
        TOP_REASON=$(echo "$LIVE_REJECTED" | head -1 | sed 's/|/ × /')
        REJECT_SUMMARY="
${LIVE_SIGNAL_COUNT} signal(s) generated but all rejected:
  ${TOP_REASON}"
    elif [[ "$LIVE_SIGNAL_COUNT" -eq 0 ]]; then
        REJECT_SUMMARY="
No signals generated (filters active)."
    fi

    LIVE_SECTION="✅ *LIVE PAPER TRADES*
No live trades today.${REJECT_SUMMARY}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: REPLAY BACKTEST — hypothetical results from replay engine
# ══════════════════════════════════════════════════════════════════════════════
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

# Parse replay output
if [[ "$REPLAY_TIMED_OUT" == "true" ]]; then
    REPLAY_SECTION="📊 *REPLAY BACKTEST* _(hypothetical)_
⚠️ Replay timed out after 5 minutes.
Run manually: \`python scripts/replay_day.py --date ${TODAY}\`"
else
    SIGNALS=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Signals:' | awk '{print $NF}' || echo "0")
    WINS=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Wins:' | awk '{print $NF}' || echo "0")
    LOSSES=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Losses:' | awk '{print $NF}' || echo "0")
    WIN_RATE=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Win Rate:' | awk '{print $NF}' || echo "0%")
    NET_PNL=$(echo "$REPLAY_OUTPUT" | grep -E '^\s+Net P&L:' | awk '{print $NF}' || echo '+$0.00')

    # Extract per-ticker lines
    TICKER_LINES=""
    while IFS= read -r line; do
        if echo "$line" | grep -qE '^\s+[0-9]+\s+[0-9]{2}:[0-9]{2}'; then
            TICKER=$(echo "$line" | awk '{print $4}')
            DIR=$(echo "$line" | awk '{print $5}')
            ENTRY_PRICE=$(echo "$line" | awk '{print $7}')
            EXIT_PRICE=$(echo "$line" | awk '{print $(NF-7)}')
            OUTCOME=$(echo "$line" | awk '{print $(NF-6)}')
            PNL=$(echo "$line" | awk '{print $(NF-3)}')
            SHARES=$(echo "$line" | awk '{print $(NF-2)}')

            if [[ "$PNL" == *"-"* ]]; then
                T_EMOJI="❌"
            else
                T_EMOJI="✅"
            fi

            if [[ "$SHARES" =~ ^[0-9]+$ ]] && [[ "$SHARES" -gt 0 ]]; then
                TICKER_LINES="${TICKER_LINES}  ${T_EMOJI} ${TICKER} ${DIR} ${SHARES}sh \$${ENTRY_PRICE}→\$${EXIT_PRICE} ${OUTCOME} (\$${PNL})"$'\n'
            else
                TICKER_LINES="${TICKER_LINES}  ${T_EMOJI} ${TICKER} ${DIR} \$${ENTRY_PRICE}→\$${EXIT_PRICE} ${OUTCOME} (\$${PNL})"$'\n'
            fi
        fi
    done <<< "$REPLAY_OUTPUT"

    if [[ "$SIGNALS" == "0" ]] || [[ -z "$SIGNALS" ]]; then
        REPLAY_SECTION="📊 *REPLAY BACKTEST* _(hypothetical)_
No replay signals for ${TODAY}."
    else
        REPLAY_SECTION="📊 *REPLAY BACKTEST* _(hypothetical)_
⚠️ These are backtest results, NOT live trades.
The replay engine uses different filters than the live scanner.

• Signals: ${SIGNALS} | Wins: ${WINS} | Losses: ${LOSSES}
• Win Rate: ${WIN_RATE} | Net P&L: ${NET_PNL}
${TICKER_LINES}"
    fi
fi

# ── Build final Telegram message ─────────────────────────────────────────────
DAY_NAME=$(date -j -f "%Y-%m-%d" "$TODAY" "+%A" 2>/dev/null || date -d "$TODAY" "+%A" 2>/dev/null || echo "")

# Overall emoji: based on live trades, not replay
if [[ "$LIVE_APPROVED_COUNT" -gt 0 ]]; then
    DAY_EMOJI="📈"
else
    DAY_EMOJI="📋"
fi

MESSAGE="${DAY_EMOJI} *ORB Daily Report — ${DAY_NAME} ${TODAY}*

${LIVE_SECTION}

─────────────────────────

${REPLAY_SECTION}"

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
