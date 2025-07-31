# Use the official Python 3.13 slim image as a base
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by rasterio and ffmpeg for GIF creation
RUN apt-get update && apt-get install -y \
    libexpat1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy only the dependency files first
COPY pyproject.toml uv.lock* ./

# Install dependencies. This layer is only rebuilt if the dependency files change.
RUN uv sync

# Copy the rest of the workflow code. This layer is rebuilt on every code change.
COPY . .

# Set the command to run the task runner
CMD ["uv", "run", "--", "python", "main.py"]