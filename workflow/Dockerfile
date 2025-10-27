FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y libexpat1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy application code (this layer rebuilds on code changes)
COPY . .

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Set the command to run the task runner
CMD ["uv", "run", "--", "python", "-m", "vci_workflow.runner"]
