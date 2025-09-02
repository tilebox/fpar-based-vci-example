FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies in a separate layer for better caching
RUN apt-get update && apt-get install -y \
    libexpat1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# Sync dependencies
WORKDIR /app
# Copy only dependency files for maximum cache efficiency
COPY pyproject.toml uv.lock ./

# Install dependencies with uv cache for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-local

# Copy our code 
WORKDIR /app

# Copy application code (this layer rebuilds on code changes)
COPY . .

RUN uv sync

# Set the command to run the task runner
CMD ["uv", "run", "--", "python", "-m", "vci_workflow.runner"]
