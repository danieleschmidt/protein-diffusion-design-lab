# Multi-stage Dockerfile for protein-diffusion-design-lab
# Optimized for both development and production use

# Build stage
FROM pytorch/pytorch:2.0.1-cuda11.7-devel as builder

LABEL maintainer="Daniel Schmidt <your.email@example.com>"
LABEL description="Protein Diffusion Design Lab - AI-powered protein engineering"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r protein && useradd -r -g protein -m -s /bin/bash protein

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/
COPY README.md LICENSE ./

# Install package in development mode
RUN pip install -e .

# Production stage
FROM pytorch/pytorch:2.0.1-cuda11.7-runtime as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PROTEIN_DIFFUSION_ENV=production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r protein && useradd -r -g protein -m -s /bin/bash protein

# Set working directory
WORKDIR /app

# Copy from builder stage
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app /app

# Create necessary directories
RUN mkdir -p /app/data /app/weights /app/outputs /app/logs \
    && chown -R protein:protein /app

# Switch to non-root user
USER protein

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import protein_diffusion; print('OK')" || exit 1

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]

# Development stage
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install pre-commit hooks
RUN git config --global --add safe.directory /app
COPY .pre-commit-config.yaml ./
RUN pre-commit install-hooks || true

# Set development environment
ENV PROTEIN_DIFFUSION_ENV=development
ENV PROTEIN_DIFFUSION_LOG_LEVEL=DEBUG

# Switch to non-root user
USER protein

# Default command for development
CMD ["bash"]