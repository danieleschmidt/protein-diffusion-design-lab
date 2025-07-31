# Multi-stage Dockerfile for protein-diffusion-design-lab
# Optimized for ML workloads with CUDA support

FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash protein_user
WORKDIR /app
RUN chown protein_user:protein_user /app

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=protein_user:protein_user . .

# Install package in development mode
RUN pip3 install -e .

# Switch to non-root user
USER protein_user

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]