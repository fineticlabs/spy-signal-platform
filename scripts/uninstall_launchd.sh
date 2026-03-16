#!/usr/bin/env bash
# ── uninstall_launchd.sh ──────────────────────────────────────────────────────
# Unload and remove the two spy-scanner launchd plists.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
START_PLIST="$LAUNCH_AGENTS_DIR/com.fineticlabs.spy-scanner-start.plist"
STOP_PLIST="$LAUNCH_AGENTS_DIR/com.fineticlabs.spy-scanner-stop.plist"

echo "Uninstalling spy-scanner launchd jobs..."
echo ""

# ── Unload start job ─────────────────────────────────────────────────────────
if [[ -f "$START_PLIST" ]]; then
    launchctl unload "$START_PLIST" 2>/dev/null || true
    rm -f "$START_PLIST"
    echo "✓ Removed: com.fineticlabs.spy-scanner-start"
else
    echo "  Start plist not found (already removed?)."
fi

# ── Unload stop job ──────────────────────────────────────────────────────────
if [[ -f "$STOP_PLIST" ]]; then
    launchctl unload "$STOP_PLIST" 2>/dev/null || true
    rm -f "$STOP_PLIST"
    echo "✓ Removed: com.fineticlabs.spy-scanner-stop"
else
    echo "  Stop plist not found (already removed?)."
fi

echo ""
echo "Done. Verify with:"
echo "  launchctl list | grep fineticlabs"
