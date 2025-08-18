# 🔐 Multi-stage Docker build with secret management
FROM python:3.11-slim as secret-builder

# Install secret manager dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt cryptography click

# Copy secret manager code
COPY config/secret_manager.py config/
COPY scripts/secret_manager_cli.py scripts/

# 🔐 Runtime stage with secrets
FROM python:3.11-slim as runtime

# Install runtime dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy secret manager from builder
COPY --from=secret-builder /app/config/secret_manager.py config/
COPY --from=secret-builder /app/scripts/secret_manager_cli.py scripts/

# 🔧 Environment setup script
RUN echo '#!/bin/bash\n\
# 🔐 Container secret initialization\n\
echo "🚀 Initializing secrets in container..."\n\
\n\
# Check if secrets are mounted or provided via env vars\n\
if [ -f "/run/secrets/app-secrets" ]; then\n\
    echo "📁 Loading secrets from Docker secrets mount"\n\
    export $(cat /run/secrets/app-secrets | xargs)\n\
elif [ -f "/app/secrets/.env" ]; then\n\
    echo "📁 Loading secrets from mounted .env"\n\
    export $(cat /app/secrets/.env | xargs)\n\
elif [ ! -z "$SECRETS_JSON" ]; then\n\
    echo "📁 Loading secrets from JSON environment variable"\n\
    echo "$SECRETS_JSON" | python3 -c "\n\
import json, sys, os\n\
secrets = json.load(sys.stdin)\n\
for k, v in secrets.items():\n\
    os.environ[k] = v\n\
    print(f\\"export {k}={v}\\")\n\
" > /tmp/secrets.env\n\
    export $(cat /tmp/secrets.env | xargs)\n\
    rm /tmp/secrets.env\n\
else\n\
    echo "⚠️  No secrets found, using defaults"\n\
fi\n\
\n\
# Validate required secrets\n\
python3 scripts/secret_manager_cli.py validate --required GROK4_API_KEY --required DATABASE_URL\n\
\n\
echo "✅ Secret initialization complete"\n\
exec "$@"\n\
' > /app/init-secrets.sh && chmod +x /app/init-secrets.sh

# 🚀 Default entrypoint with secret initialization
ENTRYPOINT ["/app/init-secrets.sh"]
CMD ["python", "app.py"]

# 📡 Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 🏷️ Labels for secret management
LABEL secret-manager="enabled"
LABEL secret-categories="ai,trading,database,security,monitoring"
LABEL deployment-ready="true"
