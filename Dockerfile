# Multi-stage Dockerfile for Intelligent Knowledge Assistant
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:$PATH"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --only main

# Copy application source code and documents
COPY src/ ./src/
COPY data/ ./data/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY README.md .env.example ./

# Ingest knowledge corpus during build
RUN poetry run python scripts/ingest.py

EXPOSE 8000 8501

# Default command starts the API service
CMD ["poetry", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
