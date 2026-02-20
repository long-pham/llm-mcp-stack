#!/usr/bin/env bash
# Uninstall claude-switch and restore original state.
#
# Usage:
#   ./uninstall.sh              # Uninstall locally
#   ./uninstall.sh --local      # Uninstall locally (explicit)
#   ./uninstall.sh user@host    # Uninstall on remote machine via SSH

set -euo pipefail

TARGET_NAME="claude-switch"
CONFIG_DIR="\$HOME/.config/claude-switch"
CLAUDE_SETTINGS="\$HOME/.claude/settings.json"
CLAUDE_JSON="\$HOME/.claude/claude.json"
STASHED_OAUTH="\$HOME/.config/claude-switch/oauth.json.bak"

# Run a command locally or remotely
run() {
    if [ "$MODE" = "local" ]; then
        bash -c "$1"
    else
        ssh "$REMOTE_HOST" "$1"
    fi
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [user@host]

Uninstall claude-switch and restore original Claude Code auth state.

Arguments:
  user@host       Uninstall on remote machine via SSH

Options:
  --local, -l     Uninstall on the local machine (default if no host given)
  --help, -h      Show this help message

This script will:
  1. Restore stashed OAuth token (if any) into claude.json
  2. Remove apiBaseUrl from settings.json (if set by claude-switch)
  3. Remove the claude-switch env sourcing line from your shell rc
  4. Remove ~/.local/bin/claude-switch
  5. Remove ~/.config/claude-switch/
  6. Unset ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY from current env file

Examples:
  $(basename "$0")                    # Uninstall locally
  $(basename "$0") dev@agx.local      # Uninstall on remote AGX Orin
EOF
    exit 0
}

# Parse arguments
MODE="local"
REMOTE_HOST=""

case "${1:-}" in
    --help|-h)     usage ;;
    --local|-l)    MODE="local" ;;
    "")            MODE="local" ;;
    -*)            echo "Unknown option: $1" >&2; usage ;;
    *)             MODE="remote"; REMOTE_HOST="$1" ;;
esac

if [ "$MODE" = "local" ]; then
    echo "Uninstalling claude-switch locally ..."
else
    echo "Uninstalling claude-switch on $REMOTE_HOST ..."
fi

# 1. Restore stashed OAuth token if present
echo "  Restoring OAuth token (if stashed) ..."
run '
CLAUDE_JSON="$HOME/.claude.json"
CLAUDE_JSON_LEGACY="$HOME/.claude/claude.json"
STASHED_OAUTH="$HOME/.config/claude-switch/oauth.json.bak"
if [ -f "$STASHED_OAUTH" ] && command -v python3 &>/dev/null; then
    BAK="$STASHED_OAUTH" python3 -c "
import json, os, sys
bak = os.environ[\"BAK\"]
files = sys.argv[1:]
with open(bak) as f:
    oauth = json.load(f)
restored = False
for path in files:
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        d = json.load(f)
    d[\"oauthAccount\"] = oauth
    with open(path + \".tmp\", \"w\") as f:
        json.dump(d, f, indent=2)
        f.write(\"\\n\")
    os.replace(path + \".tmp\", path)
    restored = True
if restored:
    os.remove(bak)
    print(\"  OAuth token restored\")
" "$CLAUDE_JSON" "$CLAUDE_JSON_LEGACY" < /dev/null 2>/dev/null || true
else
    echo "  No stashed OAuth token found (nothing to restore)"
fi
'

# 2. Remove apiBaseUrl from settings.json if present
echo "  Cleaning settings.json ..."
run '
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
if [ -f "$CLAUDE_SETTINGS" ] && grep -q "apiBaseUrl" "$CLAUDE_SETTINGS" 2>/dev/null && command -v python3 &>/dev/null; then
    python3 -c "
import json, sys
with open(\"$CLAUDE_SETTINGS\") as f:
    d = json.load(f)
d.pop(\"apiBaseUrl\", None)
with open(\"$CLAUDE_SETTINGS.tmp\", \"w\") as f:
    json.dump(d, f, indent=2)
    f.write(\"\\n\")
import os; os.replace(\"$CLAUDE_SETTINGS.tmp\", \"$CLAUDE_SETTINGS\")
print(\"  Removed apiBaseUrl from settings.json\")
" < /dev/null 2>/dev/null || echo "  Failed to clean settings.json (manual edit may be needed)"
else
    echo "  settings.json clean (no apiBaseUrl found)"
fi
'

# 3. Remove claude-switch env sourcing from shell rc
echo "  Cleaning shell rc ..."
run '
if [ -f ~/.zshrc ]; then
    SHELL_RC=~/.zshrc
elif [ -f ~/.bashrc ]; then
    SHELL_RC=~/.bashrc
else
    SHELL_RC=""
fi

if [ -n "$SHELL_RC" ]; then
    cp "$SHELL_RC" "${SHELL_RC}.pre-claude-switch-uninstall"
    # Remove the comment line and the sourcing line
    sed -i.tmp "/^# Claude Code auth switcher$/d" "$SHELL_RC"
    sed -i.tmp "/claude-switch\/env/d" "$SHELL_RC"
    rm -f "${SHELL_RC}.tmp"
    echo "  Cleaned $SHELL_RC (backup: ${SHELL_RC}.pre-claude-switch-uninstall)"
else
    echo "  No shell rc found to clean"
fi
'

# 4. Remove the claude-switch binary
echo "  Removing ~/.local/bin/claude-switch ..."
run 'rm -f "$HOME/.local/bin/claude-switch"'

# 5. Remove config directory
echo "  Removing ~/.config/claude-switch/ ..."
run 'rm -rf "$HOME/.config/claude-switch"'

echo ""
echo "Done. claude-switch has been uninstalled."
echo ""
echo "To complete cleanup in your current shell, run:"
echo "  unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY"
echo "  source ~/.zshrc  # or restart your shell"
