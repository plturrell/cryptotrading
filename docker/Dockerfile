# Multi-stage Docker build for CryptoTrading Platform
FROM node:18-alpine AS frontend-builder

# Build frontend assets
WORKDIR /app
COPY package*.json ./
COPY ui5*.yaml ./
RUN npm ci --only=production

COPY webapp/ ./webapp/
RUN npm run build:prod

# Python application stage
FROM python:3.11-slim AS backend

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_CACHE_DIR=/opt/poetry-cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Create app directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY src/ ./src/
COPY app.py ./
COPY requirements.txt ./

# Copy built frontend assets
COPY --from=frontend-builder /app/dist ./webapp/

# Create data directory
RUN mkdir -p data/models data/cache logs

# Create non-root user
RUN groupadd -r cryptotrading && useradd -r -g cryptotrading cryptotrading
RUN chown -R cryptotrading:cryptotrading /app
USER cryptotrading

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["poetry", "run", "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "gevent", "--timeout", "120", "app:application"]
