# Cloud Scheduler Simulator Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY cloud_scheduler/ ./cloud_scheduler/
COPY configs/ ./configs/
COPY README.md ./

# Install Python dependencies
RUN uv sync --frozen

# Create directories
RUN mkdir -p data/{raw,processed} experiments logs notebooks

# Set default command
CMD ["uv", "run", "cloud-sim", "--help"]
