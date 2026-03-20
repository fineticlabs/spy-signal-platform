#!/usr/bin/env bash
# Graceful scanner shutdown: SIGTERM → wait → SIGKILL
set -uo pipefail
PROJECT_DIR="$HOME/Desktop/I/Projects/spy-signal-platform"
PID_FILE="$PROJECT_DIR/logs/scanner.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file — scanner not running."
    exit 0
fi

PID=$(cat "$PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
    echo "PID $PID not running. Cleaning up PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Sending SIGTERM to scanner (PID $PID)..."
kill "$PID"

for i in $(seq 1 10); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Scanner stopped gracefully after ${i}s."
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

echo "Scanner didn't stop gracefully. Sending SIGKILL..."
kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Scanner force-killed."
