# claude-switch

Toggle Claude Code between **proxy mode** (custom API endpoint) and **official mode** (Claude AI OAuth).

## Quick Start

```bash
# Install locally
./install.sh

# Or install on a remote machine via SSH
./install.sh user@hostname
```

After install, restart your shell or run:
```bash
source ~/.config/claude-switch/env
```

## Usage

```
claude-switch <command> [options]

Commands:
  proxy [URL [API_KEY]]   Switch to proxy mode
  official                Switch to official Claude AI OAuth
  status                  Show current auth mode
  config                  Show config file path and current values
```

### Switch to proxy mode

```bash
# Use saved config values
claude-switch proxy

# Override proxy URL (saved for future use)
claude-switch proxy http://myhost:8080/api/anthropic

# Override both URL and API key
claude-switch proxy http://myhost:8080/api/anthropic sk-my-api-key
```

### Switch to official OAuth

```bash
claude-switch official
```

### Check current mode

```bash
claude-switch status
```

## How It Works

`claude-switch` manages two things when switching modes:

1. **`~/.claude/settings.json`** — Sets or removes the `apiBaseUrl` field that Claude Code reads on startup.
2. **Environment variables** — Writes a sourceable file (`~/.config/claude-switch/env`) that exports or unsets `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`. This file is sourced automatically by your shell rc.

### Proxy mode
- Sets `apiBaseUrl` in `settings.json` to your proxy URL
- Exports `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`

### Official mode
- Removes `apiBaseUrl` from `settings.json`
- Unsets `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`
- Claude Code uses its built-in OAuth flow

## Configuration

Config file: `~/.config/claude-switch/config`

```
PROXY_BASE_URL=http://your-proxy-host:11212/api/anthropic
PROXY_API_KEY=your-api-key
```

Edit manually or pass values via CLI:
```bash
claude-switch proxy http://newhost:9090/api/anthropic new-key
```

CLI arguments are persisted to the config file automatically.

## Install Details

### What `install.sh` does

1. Copies `claude-switch.sh` to `~/.local/bin/claude-switch`
2. Adds `~/.local/bin` to `$PATH` in your shell rc (idempotent)
3. Adds auto-sourcing of `~/.config/claude-switch/env` to your shell rc (idempotent)
4. Removes any old hardcoded `ANTHROPIC_*` exports from your shell rc (backs up to `.bak`)
5. Runs `claude-switch status` to initialize config

### Supported targets

```bash
./install.sh                # Local machine
./install.sh dev@agx.local  # Remote via SSH
./install.sh user@server    # Any SSH-accessible host
```

Shell rc detection: prefers `.zshrc`, falls back to `.bashrc`.

### Requirements

- **bash** (4.0+)
- **python3** (for safe JSON manipulation of `settings.json`)
- **ssh/scp** (remote install only)

## File Layout

```
~/.local/bin/claude-switch           # The script
~/.config/claude-switch/
├── config                           # PROXY_BASE_URL and PROXY_API_KEY
├── env                              # Sourceable shell exports (auto-generated)
└── active                           # Current mode: "proxy" or "official"
~/.claude/settings.json              # Claude Code settings (apiBaseUrl managed here)
```
