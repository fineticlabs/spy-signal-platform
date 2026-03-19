#!/usr/bin/env bash
# ── health_check.sh ──────────────────────────────────────────────────────────
# Post-start health check for the live scanner.
# Run ~5 minutes after auto_start.sh to verify the pipeline is healthy.
#
# Checks:
#   1. Scanner process alive (PID file)
#   2. Bars flowing (bar_received in last 60s of log)
#   3. Regime data populated (regime_updated with non-None vix/adx)
#   4. No regime gate blocking (orb_filter_no_regime_data absent)
#   5. No unknown indicator warnings (snapshot_unknown_indicator absent)
#
# Exit codes:
#   0  All checks passed (GO)
#   1  One or more checks failed (NO-GO)
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PROJECT_DIR="$HOME/Desktop/I/Projects/spy-signal-platform"
LOGS_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOGS_DIR/scanner.pid"
TODAY=$(date +%Y-%m-%d)
LOG_FILE="$LOGS_DIR/scanner_${TODAY}.log"

PASS=0
FAIL=0

pass() { echo "  [PASS] $1"; ((PASS++)); }
fail() { echo "  [FAIL] $1"; echo "         -> $2"; ((FAIL++)); }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Scanner Health Check — $(date '+%Y-%m-%d %H:%M:%S %Z')  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Scanner process alive ────────────────────────────────────────────────

if [[ ! -f "$PID_FILE" ]]; then
    fail "Process alive" "PID file missing at $PID_FILE"
    echo "         Fix: bash scripts/auto_start.sh"
else
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        pass "Process alive (PID $PID)"
    else
        fail "Process alive" "PID $PID is not running"
        echo "         Fix: rm $PID_FILE && bash scripts/auto_start.sh"
    fi
fi

# ── Guard: log file must exist for remaining checks ─────────────────────────

if [[ ! -f "$LOG_FILE" ]]; then
    fail "Log file" "No log file at $LOG_FILE"
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  Result: NO-GO ($PASS passed, $FAIL failed)"
    echo "  Cannot run remaining checks without a log file."
    echo "══════════════════════════════════════════════════════════"
    exit 1
fi

# ── 2. Bars flowing (bar_received in last 60 seconds) ───────────────────────

NOW=$(date +%s)
FOUND_RECENT_BAR=false

# Read the last 200 lines looking for bar_received with a timestamp within 60s
while IFS= read -r line; do
    # Extract ISO timestamp from structured log (format: 2026-03-19 09:35:12)
    ts=$(echo "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
    if [[ -n "$ts" ]]; then
        # macOS date -j for parsing, Linux would use date -d
        line_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$ts" +%s 2>/dev/null || echo 0)
        age=$(( NOW - line_epoch ))
        if (( age >= 0 && age <= 60 )); then
            FOUND_RECENT_BAR=true
            break
        fi
    fi
done < <(grep "bar_received" "$LOG_FILE" | tail -200)

if $FOUND_RECENT_BAR; then
    pass "Bars flowing (bar_received within last 60s)"
else
    fail "Bars flowing" "No bar_received in last 60 seconds of log"
    echo "         Fix: Check Alpaca websocket connection and API keys"
    echo "         Debug: tail -50 $LOG_FILE | grep -i 'websocket\|error\|disconnect'"
fi

# ── 3. Regime data populated (non-None vix and adx) ─────────────────────────

# Look for the most recent regime_updated line
LAST_REGIME=$(grep "regime_updated" "$LOG_FILE" | tail -1)

if [[ -z "$LAST_REGIME" ]]; then
    fail "Regime populated" "No regime_updated entries in log"
    echo "         Fix: Indicators may not have warmed up yet (need ~28 bars for ADX)"
    echo "         Wait 5 more minutes and re-run this check"
else
    VIX_NONE=$(echo "$LAST_REGIME" | grep -c "vix=None" || true)
    ADX_NONE=$(echo "$LAST_REGIME" | grep -c "adx=None" || true)

    if (( VIX_NONE > 0 )) && (( ADX_NONE > 0 )); then
        fail "Regime populated" "Both vix=None and adx=None in latest regime_updated"
        echo "         Fix: Check _process_bar() is calling regime.update(vix=..., adx=...)"
    elif (( VIX_NONE > 0 )); then
        fail "Regime populated" "vix=None in latest regime_updated"
        echo "         Fix: Check _VIX_FALLBACK in src/main.py"
    elif (( ADX_NONE > 0 )); then
        fail "Regime populated" "adx=None — StreamingADX not warmed up yet"
        echo "         Fix: ADX needs ~28 bars to warm up. Wait a few more minutes."
    else
        # Extract actual values for display
        VIX_VAL=$(echo "$LAST_REGIME" | grep -oE "vix=[0-9.]+" | head -1)
        ADX_VAL=$(echo "$LAST_REGIME" | grep -oE "adx=[0-9.]+" | head -1)
        pass "Regime populated ($VIX_VAL, $ADX_VAL)"
    fi
fi

# ── 4. No regime gate blocking ──────────────────────────────────────────────

REGIME_BLOCKS=$(tail -30 "$LOG_FILE" | grep -c "orb_filter_no_regime_data" || true)

if (( REGIME_BLOCKS == 0 )); then
    pass "No regime gate blocks (orb_filter_no_regime_data absent from last 30 lines)"
else
    fail "Regime gate" "$REGIME_BLOCKS occurrences of orb_filter_no_regime_data in last 30 log lines"
    echo "         Fix: regime.update() is not receiving vix/adx values"
    echo "         Debug: grep 'regime_updated' $LOG_FILE | tail -5"
fi

# ── 5. No unknown indicator warnings (last 5 minutes only) ──────────────────

UNKNOWN_COUNT=0
UNKNOWN_NAMES_LIST=""
WINDOW_SECS=300  # 5 minutes

while IFS= read -r line; do
    ts=$(echo "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
    if [[ -n "$ts" ]]; then
        line_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$ts" +%s 2>/dev/null || echo 0)
        age=$(( NOW - line_epoch ))
        if (( age >= 0 && age <= WINDOW_SECS )); then
            ((UNKNOWN_COUNT++))
            name=$(echo "$line" | grep -oE "name=[a-z_]+" | head -1)
            UNKNOWN_NAMES_LIST="${UNKNOWN_NAMES_LIST}${name}"$'\n'
        fi
    fi
done < <(grep "snapshot_unknown_indicator" "$LOG_FILE" | tail -500)

if (( UNKNOWN_COUNT == 0 )); then
    pass "No unknown indicator warnings (last 5 min)"
else
    UNIQUE_NAMES=$(echo "$UNKNOWN_NAMES_LIST" | sort -u | tr '\n' ', ')
    fail "Unknown indicators" "$UNKNOWN_COUNT in last 5 min for: $UNIQUE_NAMES"
    echo "         Fix: Add the indicator name to _known set in src/indicators/registry.py"
fi

# ── Verdict ─────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════"
if (( FAIL == 0 )); then
    echo "  Result: GO ($PASS/$((PASS+FAIL)) checks passed)"
    echo "  Scanner is healthy. Signals should flow during market hours."
else
    echo "  Result: NO-GO ($PASS passed, $FAIL failed)"
    echo "  Fix the issues above and re-run: bash scripts/health_check.sh"
fi
echo "══════════════════════════════════════════════════════════"
echo ""

(( FAIL == 0 ))
