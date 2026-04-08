# Use official Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create non-root user for security
RUN useradd -m openenv && chown -R openenv:openenv /app
USER openenv

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from env.environment import MyOpenEnvEnvironment; env = MyOpenEnvEnvironment(); env.reset()" || exit 1

# Default command
CMD ["python", "inference.py"]
