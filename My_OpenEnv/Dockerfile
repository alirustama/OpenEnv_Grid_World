# Use official Python runtime
FROM python:3.11-slim

# Application directory
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Security: create non-root user
RUN useradd -m openenv && chown -R openenv:openenv /app
USER openenv

# Port for optional Gradio UI
EXPOSE 7860

# Healthcheck ensures environment can initialize
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from env.environment import MyOpenEnvEnvironment; env = MyOpenEnvEnvironment(); env.reset()" || exit 1

# Default entrypoint
CMD ["python", "inference.py"]
