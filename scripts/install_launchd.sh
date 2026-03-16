#!/usr/bin/env bash
# ── install_launchd.sh ────────────────────────────────────────────────────────
# Create and load two launchd plists for automated market-hours scheduling:
#   1. com.fineticlabs.spy-scanner-start  → 6:25 AM PT Mon–Fri
#   2. com.fineticlabs.spy-scanner-stop   → 1:05 PM PT Mon–Fri
#
# NOTE: macOS must be awake for launchd jobs to fire. If your Mac sleeps,
# consider using `caffeinate` or adjusting Energy Saver settings to prevent
# sleep during market hours (6:20 AM – 1:10 PM PT).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$HOME/Desktop/I/Projects/spy-signal-platform"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
START_PLIST="$LAUNCH_AGENTS_DIR/com.fineticlabs.spy-scanner-start.plist"
STOP_PLIST="$LAUNCH_AGENTS_DIR/com.fineticlabs.spy-scanner-stop.plist"

echo "Installing launchd jobs for spy-signal-platform..."
echo ""

# ── Validate scripts exist ────────────────────────────────────────────────────
for script in auto_start.sh auto_stop.sh; do
    if [[ ! -f "$PROJECT_DIR/scripts/$script" ]]; then
        echo "ERROR: $PROJECT_DIR/scripts/$script not found."
        exit 1
    fi
done

# ── Make scripts executable ───────────────────────────────────────────────────
chmod +x "$PROJECT_DIR/scripts/auto_start.sh"
chmod +x "$PROJECT_DIR/scripts/auto_stop.sh"
echo "✓ Scripts marked executable."

# ── Ensure LaunchAgents directory exists ──────────────────────────────────────
mkdir -p "$LAUNCH_AGENTS_DIR"

# ── Unload existing jobs (if any) ─────────────────────────────────────────────
if launchctl list | grep -q "com.fineticlabs.spy-scanner-start"; then
    launchctl unload "$START_PLIST" 2>/dev/null || true
    echo "  Unloaded existing start job."
fi
if launchctl list | grep -q "com.fineticlabs.spy-scanner-stop"; then
    launchctl unload "$STOP_PLIST" 2>/dev/null || true
    echo "  Unloaded existing stop job."
fi

# ── Create start plist (6:25 AM PT = 13:25 UTC standard / 13:25 UTC) ─────────
# We use the system's local timezone. Since the Mac is in PT, hour/minute
# are specified in local time.
cat > "$START_PLIST" << 'PLIST_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.fineticlabs.spy-scanner-start</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>REPLACE_PROJECT_DIR/scripts/auto_start.sh</string>
    </array>

    <key>StartCalendarInterval</key>
    <array>
        <!-- Monday -->
        <dict>
            <key>Weekday</key><integer>1</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>25</integer>
        </dict>
        <!-- Tuesday -->
        <dict>
            <key>Weekday</key><integer>2</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>25</integer>
        </dict>
        <!-- Wednesday -->
        <dict>
            <key>Weekday</key><integer>3</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>25</integer>
        </dict>
        <!-- Thursday -->
        <dict>
            <key>Weekday</key><integer>4</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>25</integer>
        </dict>
        <!-- Friday -->
        <dict>
            <key>Weekday</key><integer>5</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>25</integer>
        </dict>
    </array>

    <key>StandardOutPath</key>
    <string>REPLACE_PROJECT_DIR/logs/launchd_start.log</string>
    <key>StandardErrorPath</key>
    <string>REPLACE_PROJECT_DIR/logs/launchd_start.log</string>

    <key>WorkingDirectory</key>
    <string>REPLACE_PROJECT_DIR</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
PLIST_EOF

# ── Create stop plist (1:05 PM PT) ───────────────────────────────────────────
cat > "$STOP_PLIST" << 'PLIST_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.fineticlabs.spy-scanner-stop</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>REPLACE_PROJECT_DIR/scripts/auto_stop.sh</string>
    </array>

    <key>StartCalendarInterval</key>
    <array>
        <!-- Monday -->
        <dict>
            <key>Weekday</key><integer>1</integer>
            <key>Hour</key><integer>13</integer>
            <key>Minute</key><integer>5</integer>
        </dict>
        <!-- Tuesday -->
        <dict>
            <key>Weekday</key><integer>2</integer>
            <key>Hour</key><integer>13</integer>
            <key>Minute</key><integer>5</integer>
        </dict>
        <!-- Wednesday -->
        <dict>
            <key>Weekday</key><integer>3</integer>
            <key>Hour</key><integer>13</integer>
            <key>Minute</key><integer>5</integer>
        </dict>
        <!-- Thursday -->
        <dict>
            <key>Weekday</key><integer>4</integer>
            <key>Hour</key><integer>13</integer>
            <key>Minute</key><integer>5</integer>
        </dict>
        <!-- Friday -->
        <dict>
            <key>Weekday</key><integer>5</integer>
            <key>Hour</key><integer>13</integer>
            <key>Minute</key><integer>5</integer>
        </dict>
    </array>

    <key>StandardOutPath</key>
    <string>REPLACE_PROJECT_DIR/logs/launchd_stop.log</string>
    <key>StandardErrorPath</key>
    <string>REPLACE_PROJECT_DIR/logs/launchd_stop.log</string>

    <key>WorkingDirectory</key>
    <string>REPLACE_PROJECT_DIR</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
PLIST_EOF

# ── Replace placeholder with actual project path ─────────────────────────────
sed -i '' "s|REPLACE_PROJECT_DIR|$PROJECT_DIR|g" "$START_PLIST"
sed -i '' "s|REPLACE_PROJECT_DIR|$PROJECT_DIR|g" "$STOP_PLIST"

echo "✓ Plists created:"
echo "  $START_PLIST"
echo "  $STOP_PLIST"

# ── Load the jobs ─────────────────────────────────────────────────────────────
launchctl load "$START_PLIST"
launchctl load "$STOP_PLIST"

echo "✓ Jobs loaded into launchd."
echo ""
echo "Schedule:"
echo "  START: 6:25 AM PT Mon–Fri  (backfill + launch scanner)"
echo "  STOP:  1:05 PM PT Mon–Fri  (kill scanner + replay + Telegram summary)"
echo ""
echo "Verify with:"
echo "  launchctl list | grep fineticlabs"
echo ""
echo "⚠️  IMPORTANT: Your Mac must be awake for these jobs to fire."
echo "   To prevent sleep during market hours, run:"
echo "   caffeinate -s -w \$\$ &"
echo "   Or adjust System Settings → Energy Saver → Prevent automatic sleeping."
