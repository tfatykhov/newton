FROM python:3.12-slim

WORKDIR /app

# Install psql client, curl, and Node.js (required by Claude Agent SDK's bundled CLI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Create SDK workspace directory
RUN mkdir -p /tmp/nous-workspace

# Copy source BEFORE pip install (F5: non-editable install)
COPY pyproject.toml .
COPY nous/ nous/
RUN pip install --no-cache-dir ".[runtime]"

COPY sql/ sql/

HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "nous.main"]
