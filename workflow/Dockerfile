# Multi-stage build for better caching and smaller final image
FROM python:3.13-slim as base

# Install system dependencies in a separate layer for better caching
RUN apt-get update && apt-get install -y \
    libexpat1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv in base layer
RUN pip install --no-cache-dir uv

# Dependencies stage - this will be cached unless dependencies change
FROM base as deps
WORKDIR /app

# Copy only dependency files for maximum cache efficiency
COPY pyproject.toml uv.lock ./

# Install dependencies with uv cache for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Final stage - copy code and set up runtime
FROM deps as runtime
WORKDIR /app

# Copy application code (this layer rebuilds on code changes)
COPY . .

# Set the command to run the task runner
CMD ["uv", "run", "--", "python", "runner.py"]