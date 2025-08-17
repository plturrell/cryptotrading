FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mcp
WORKDIR /app
RUN chown mcp:mcp /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .
RUN chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Create config directory
RUN mkdir -p /app/config /app/logs

# Default environment variables
ENV MCP_TRANSPORT=http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8080
ENV MCP_REQUIRE_AUTH=true
ENV MCP_RATE_LIMIT_GLOBAL=1000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${MCP_PORT}/mcp/status || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-m", "cryptotrading.core.protocols.mcp.quick_start", \
     "--template", "prod", "--transport", "http", "--port", "8080"]

# Development variant
FROM base as development

USER root
RUN pip install pytest pytest-cov black isort mypy
USER mcp

ENV MCP_REQUIRE_AUTH=false
ENV MCP_DEBUG_LOGGING=true

CMD ["python", "-m", "cryptotrading.core.protocols.mcp.quick_start", \
     "--template", "dev", "--transport", "http", "--port", "8080"]

# Production variant
FROM base as production

# Additional security hardening
USER root
RUN rm -rf /var/cache/apt /tmp/* /var/tmp/*
USER mcp

# Stricter settings
ENV MCP_REQUIRE_AUTH=true
ENV MCP_STRICT_VALIDATION=true
ENV MCP_AUDIT_LOGGING=true

CMD ["python", "-m", "cryptotrading.core.protocols.mcp.quick_start", \
     "--template", "prod", "--transport", "http", "--port", "8080"]