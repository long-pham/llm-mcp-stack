#!/usr/bin/env bash
# Install claude-switch locally or on a remote machine via SSH.
#
# Usage:
#   ./install.sh              # Install locally
#   ./install.sh --local      # Install locally (explicit)
#   ./install.sh user@host    # Install on remote machine via SSH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_SRC="$SCRIPT_DIR/claude-switch.sh"
TARGET_NAME="claude-switch"

# Run a command locally or remotely
run() {
    if [ "$MODE" = "local" ]; then
        bash -c "$1"
    else
        ssh "$REMOTE_HOST" "$1"
    fi
}

# Copy a file to the install destination
copy_file() {
    local src="$1"
    if [ "$MODE" = "local" ]; then
        cp "$src" "$HOME/.local/bin/$TARGET_NAME"
    else
        # Use ~ so scp expands to the remote user's home, not local $HOME
        scp "$src" "$REMOTE_HOST:~/.local/bin/$TARGET_NAME"
    fi
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [user@host]

Install claude-switch for Claude Code auth mode switching.

Arguments:
  user@host       Install on remote machine via SSH (e.g., dev@agx.local)

Options:
  --local, -l     Install on the local machine (default if no host given)
  --help, -h      Show this help message

Examples:
  $(basename "$0")                    # Install locally
  $(basename "$0") --local            # Install locally (explicit)
  $(basename "$0") dev@agx.local      # Install on remote AGX Orin
  $(basename "$0") user@server.lan    # Install on any SSH-accessible machine
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

if [ ! -f "$SCRIPT_SRC" ]; then
    echo "ERROR: Source script not found: $SCRIPT_SRC" >&2
    exit 1
fi

if [ "$MODE" = "local" ]; then
    echo "Installing claude-switch locally ..."
else
    echo "Installing claude-switch on $REMOTE_HOST ..."
fi

# 1. Copy the script to ~/.local/bin
run "mkdir -p ~/.local/bin"
copy_file "$SCRIPT_SRC"
run "chmod +x ~/.local/bin/$TARGET_NAME"

# 2. Detect shell rc file on the target
SHELL_RC=$(run 'if [ -f ~/.zshrc ]; then echo ~/.zshrc; elif [ -f ~/.bashrc ]; then echo ~/.bashrc; else echo ~/.zshrc; fi')
echo "  Shell config: $SHELL_RC"

# 3. Ensure ~/.local/bin is on PATH (idempotent)
run "grep -q '\.local/bin' $SHELL_RC 2>/dev/null || echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> $SHELL_RC"

# 4. Add env sourcing to shell rc (idempotent)
run "grep -q 'claude-switch/env' $SHELL_RC 2>/dev/null || printf '\n# Claude Code auth switcher\n[ -f ~/.config/claude-switch/env ] && source ~/.config/claude-switch/env\n' >> $SHELL_RC"

# 5. Remove any old hardcoded ANTHROPIC exports (with backup)
run "cp $SHELL_RC ${SHELL_RC}.bak && sed -i.tmp '/^export ANTHROPIC_BASE_URL=/d; /^export ANTHROPIC_API_KEY=/d' $SHELL_RC && rm -f ${SHELL_RC}.tmp 2>/dev/null || true"

# 6. Initialize config and show status
run "~/.local/bin/$TARGET_NAME status"

echo ""
if [ "$MODE" = "local" ]; then
    echo "Done. Run 'claude-switch status' to verify."
else
    echo "Done. SSH into $REMOTE_HOST and run 'claude-switch status' to verify."
fi
echo "Use 'claude-switch proxy' or 'claude-switch official' to switch modes."
