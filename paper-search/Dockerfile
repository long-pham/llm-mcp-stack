FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir academic-mcp

RUN mkdir -p /data/papers

ENV ACADEMIC_MCP_DOWNLOAD_PATH=/data/papers

EXPOSE 8000

CMD ["academic-mcp", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
