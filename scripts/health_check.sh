#!/usr/bin/env bash
# ── health_check.sh ──────────────────────────────────────────────────────────
# Post-start health check for the live scanner.
# Adapts checks based on time of day:
#   PRE-MARKET (before 9:30 ET): verify websocket connected + subscribed
#   MARKET HOURS (9:30-16:00 ET): verify bars flowing + regime + signals
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

# ── Determine market phase ───────────────────────────────────────────────────
# Get current ET hour:minute (macOS compatible)
ET_TIME=$(TZ="America/New_York" date '+%H:%M')
ET_HOUR=$(echo "$ET_TIME" | cut -d: -f1 | sed 's/^0//')
ET_MIN=$(echo "$ET_TIME" | cut -d: -f2 | sed 's/^0//')
ET_TOTAL_MIN=$(( ET_HOUR * 60 + ET_MIN ))

MARKET_OPEN_MIN=$(( 9 * 60 + 30 ))   # 9:30 ET
MARKET_CLOSE_MIN=$(( 16 * 60 ))       # 16:00 ET

if (( ET_TOTAL_MIN < MARKET_OPEN_MIN )); then
    PHASE="PRE-MARKET"
elif (( ET_TOTAL_MIN <= MARKET_CLOSE_MIN )); then
    PHASE="MARKET"
else
    PHASE="AFTER-HOURS"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Scanner Health Check — $(date '+%Y-%m-%d %H:%M:%S %Z')  ║"
echo "║  Phase: ${PHASE} (${ET_TIME} ET)                            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

NOW=$(date +%s)
WINDOW_SECS=300  # 5 minutes

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

# ── 2. Websocket connected (all phases) ─────────────────────────────────────

WS_AUTH_FOUND=false
while IFS= read -r line; do
    ts=$(echo "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
    if [[ -n "$ts" ]]; then
        line_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$ts" +%s 2>/dev/null || echo 0)
        age=$(( NOW - line_epoch ))
        if (( age >= 0 && age <= WINDOW_SECS )); then
            WS_AUTH_FOUND=true
            break
        fi
    fi
done < <(grep "websocket_authenticated" "$LOG_FILE" | tail -50)

# Also accept if it was authenticated earlier today (not just last 5 min)
if ! $WS_AUTH_FOUND; then
    if grep -q "websocket_authenticated" "$LOG_FILE" 2>/dev/null; then
        WS_AUTH_FOUND=true
    fi
fi

if $WS_AUTH_FOUND; then
    pass "Websocket authenticated"
else
    fail "Websocket authenticated" "No websocket_authenticated in today's log"
    echo "         Fix: Websocket never connected. Check API keys and network."
    echo "         Debug: grep -i 'websocket\|connect\|auth' $LOG_FILE | tail -10"
fi

# ── 3. Websocket subscribed (all phases) ────────────────────────────────────

if grep -q "websocket_subscribed" "$LOG_FILE" 2>/dev/null; then
    SYMBOL_COUNT=$(grep "websocket_subscribed" "$LOG_FILE" | tail -1 | grep -oE "count=[0-9]+" | cut -d= -f2 || echo "?")
    pass "Websocket subscribed ($SYMBOL_COUNT symbols)"
else
    fail "Websocket subscribed" "No websocket_subscribed in today's log"
    echo "         Fix: Auth may have succeeded but subscription failed."
    echo "         Debug: grep 'websocket' $LOG_FILE | tail -10"
fi

# ── 4. No websocket connection errors (last 5 minutes) ──────────────────────

WS_ERROR_COUNT=0
WS_ERROR_PATTERNS="connection limit exceeded|Errno 49|Can't assign requested address|no close frame|websocket_connect_failed|websocket_retries_exhausted|websocket_startup_timeout"

while IFS= read -r line; do
    ts=$(echo "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
    if [[ -n "$ts" ]]; then
        line_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$ts" +%s 2>/dev/null || echo 0)
        age=$(( NOW - line_epoch ))
        if (( age >= 0 && age <= WINDOW_SECS )); then
            ((WS_ERROR_COUNT++))
        fi
    fi
done < <(grep -iE "$WS_ERROR_PATTERNS" "$LOG_FILE" | tail -200)

if (( WS_ERROR_COUNT == 0 )); then
    pass "No websocket errors (last 5 min)"
else
    fail "Websocket errors" "$WS_ERROR_COUNT connection errors in last 5 min"
    echo "         Fix: Kill scanner, wait 30s, restart: bash scripts/auto_start.sh"
    echo "         If 'connection limit exceeded': wait 60s for Alpaca to release connections"
    echo "         Debug: grep -iE 'websocket|Errno|connection' $LOG_FILE | tail -10"
fi

# ══════════════════════════════════════════════════════════════════════════════
# MARKET-HOURS-ONLY checks (skip during pre-market and after-hours)
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$PHASE" == "MARKET" ]]; then

    # ── 5. Bars flowing (bar_received in last 60 seconds) ────────────────────

    FOUND_RECENT_BAR=false
    while IFS= read -r line; do
        ts=$(echo "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
        if [[ -n "$ts" ]]; then
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

    # ── 6. Regime data populated (non-None vix and adx) ──────────────────────

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
            VIX_VAL=$(echo "$LAST_REGIME" | grep -oE "vix=[0-9.]+" | head -1)
            ADX_VAL=$(echo "$LAST_REGIME" | grep -oE "adx=[0-9.]+" | head -1)
            pass "Regime populated ($VIX_VAL, $ADX_VAL)"
        fi
    fi

    # ── 7. No regime gate blocking ───────────────────────────────────────────

    REGIME_BLOCKS=$(tail -30 "$LOG_FILE" | grep -c "orb_filter_no_regime_data" || true)

    if (( REGIME_BLOCKS == 0 )); then
        pass "No regime gate blocks (orb_filter_no_regime_data absent from last 30 lines)"
    else
        fail "Regime gate" "$REGIME_BLOCKS occurrences of orb_filter_no_regime_data in last 30 log lines"
        echo "         Fix: regime.update() is not receiving vix/adx values"
        echo "         Debug: grep 'regime_updated' $LOG_FILE | tail -5"
    fi

    # ── 8. No unknown indicator warnings (last 5 minutes only) ───────────────

    UNKNOWN_COUNT=0
    UNKNOWN_NAMES_LIST=""

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

else
    # Pre-market or after-hours — skip market-specific checks
    echo "  [SKIP] Bars flowing — not during market hours ($PHASE)"
    echo "  [SKIP] Regime populated — not during market hours ($PHASE)"
    echo "  [SKIP] Regime gate — not during market hours ($PHASE)"
    echo "  [SKIP] Unknown indicators — not during market hours ($PHASE)"
fi

# ── Verdict ─────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════"
if (( FAIL == 0 )); then
    if [[ "$PHASE" == "PRE-MARKET" ]]; then
        echo "  Result: GO ($PASS/$((PASS+FAIL)) checks passed)"
        echo "  PRE-MARKET: Websocket connected, waiting for market open at 9:30 ET."
    elif [[ "$PHASE" == "MARKET" ]]; then
        echo "  Result: GO ($PASS/$((PASS+FAIL)) checks passed)"
        echo "  Scanner is healthy. Signals should flow during market hours."
    else
        echo "  Result: GO ($PASS/$((PASS+FAIL)) checks passed)"
        echo "  AFTER-HOURS: Scanner connected, market closed."
    fi
else
    echo "  Result: NO-GO ($PASS passed, $FAIL failed)"
    echo "  Fix the issues above and re-run: bash scripts/health_check.sh"
fi
echo "══════════════════════════════════════════════════════════"
echo ""

(( FAIL == 0 ))
