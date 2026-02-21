# Build stage
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --prefix=/install academic-mcp

# Final stage
FROM python:3.12-slim

COPY --from=builder /install /usr/local

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root user
RUN useradd -m -u 1000 mcpuser
WORKDIR /app
RUN mkdir -p /data/papers && chown -R mcpuser:mcpuser /data/papers

USER mcpuser

ENV ACADEMIC_MCP_DOWNLOAD_PATH=/data/papers

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
  CMD curl -sf http://localhost:8000/sse >/dev/null 2>&1 || exit 1

CMD ["academic-mcp", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
