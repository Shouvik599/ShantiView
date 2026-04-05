# ShantiView - Multi-stage Docker build
# Stage 1: Build React frontend
FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json frontend/pnpm-lock.yaml* ./
RUN corepack enable && pnpm install --frozen-lockfile

COPY frontend/ .
RUN pnpm build

# Stage 2: Python backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/pyproject.toml backend/uv.lock* ./
# Grab the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Create necessary directories
RUN mkdir -p ./uploads ./models

# Copy backend code
COPY backend/ .

# Copy model files (MLP model and scaler)
COPY models/ ./models/

# Copy frontend build
COPY --from=frontend-build /app/frontend/dist ./static

ENV PORT=7860
EXPOSE 7860

# Use PORT environment variable (defaults to 7860 for Hugging Face Space compatibility)
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]