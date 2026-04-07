# Smart Hospital Resource Orchestration - Docker Configuration
# Optimized multi-stage build for production deployment

# Stage 1: Build stage with dependencies
FROM python:3.10-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set environment variables for reproducible builds
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.10-slim as runtime

# Set labels for metadata
LABEL maintainer="Smart Hospital Systems Team" \
      org.opencontainers.image.title="Smart Hospital Orchestration" \
      org.opencontainers.image.description="Advanced hospital resource management environment for reinforcement learning" \
      org.opencontainers.image.version="${VERSION:-1.0.0}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/smart-hospital/orchestration"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash hospital && \
    chown -R hospital:hospital /app

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=hospital:hospital . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R hospital:hospital /app

# Switch to non-root user
USER hospital

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import smart_hospital_orchestration; print('System healthy')" || exit 1

# Expose port for optional web interface
EXPOSE 5000

# Default command - run OpenEnv inference entrypoint (frontend-independent)
CMD ["python", "inference.py", "--task", "medium", "--seed", "42"]
