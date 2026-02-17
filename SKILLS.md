# SKILLS.md

Skills and patterns for working with the LLM MCP Stack development environment.

## OrbStack Docker Commands

The development environment runs inside an **OrbStack Linux VM** on macOS. Docker runs on the Mac host, not inside the VM. To execute Docker commands, prefix them with `mac`:

### Basic Pattern

```bash
# Run any Mac host command from inside the VM
mac <command>

# Examples
mac docker ps
mac docker compose ps
mac hostname
mac uname -a
```

### Docker Compose

The project directory must be referenced via the OrbStack mount path:

```bash
# The VM path /home/dev/llm-mcp-stack maps to:
#   /Users/<user>/OrbStack/<vm-name>/home/dev/llm-mcp-stack

# Start services
mac docker compose --project-directory /Users/lpham/OrbStack/armbian-native/home/dev/llm-mcp-stack up -d

# Check status
mac docker compose --project-directory /Users/lpham/OrbStack/armbian-native/home/dev/llm-mcp-stack ps

# View logs
mac docker compose --project-directory /Users/lpham/OrbStack/armbian-native/home/dev/llm-mcp-stack logs -f

# Stop services
mac docker compose --project-directory /Users/lpham/OrbStack/armbian-native/home/dev/llm-mcp-stack down

# Restart a single service
mac docker restart <container-name>
```

### Container Management

```bash
# View container logs
mac docker logs <container-name> --tail 30

# Inspect container
mac docker inspect <container-name>

# Stop/remove containers
mac docker stop <container-name>
mac docker rm <container-name>
```

### Network Access from VM to Docker Containers

Docker containers expose ports on the Mac host. From inside the OrbStack VM, use `host.docker.internal` to reach them:

```bash
# Test endpoints from inside the VM
curl http://host.docker.internal:3000       # MCPHub
curl http://host.docker.internal:38080      # SearXNG
curl http://host.docker.internal:38081      # SearXNG MCP
curl http://host.docker.internal:11235      # Crawl4AI
```

### Getting Host Info

```bash
mac hostname                    # Mac hostname
mac ipconfig getifaddr en0      # Mac IP address
```

### Running Integration Tests Against Docker Services

Set `MCP_BASE_HOST=host.docker.internal` to point tests at Docker containers:

```bash
MCP_BASE_HOST=host.docker.internal uv run pytest tests/mcp/test_mcp_services.py -v
MCP_BASE_HOST=host.docker.internal uv run pytest tests/mcp/test_mcp_tools.py -v
```

### Important Notes

- `mac docker exec` has path mapping issues with OrbStack — the VM filesystem paths get translated through OrbStack's mount, which can break commands that reference container-internal paths. Prefer `mac docker logs` and `mac docker inspect` instead.
- Files created in the VM at `/home/dev/llm-mcp-stack/` are visible to the Mac host at the OrbStack mount path and vice versa.
- The `.env` file must exist before running `docker compose up`. Use `start.sh` or copy from `.env.example`.
