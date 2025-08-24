#!/bin/bash

# Production startup script for Protein Diffusion Design Lab
# Terragon Labs - Autonomous SDLC v4.0

set -e

echo "🧬 Starting Protein Diffusion Design Lab - Production v4.0"
echo "📅 $(date)"
echo "🏗️  Build Environment: ${BUILD_ENV:-production}"
echo "🔢 Version: ${VERSION:-v4.0.0}"

# Configuration validation
echo "🔧 Validating configuration..."

if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL environment variable is required"
    exit 1
fi

if [ -z "$REDIS_URL" ]; then
    echo "❌ ERROR: REDIS_URL environment variable is required"
    exit 1
fi

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for PostgreSQL
echo "🐘 Waiting for PostgreSQL..."
while ! pg_isready -h $(echo $DATABASE_URL | cut -d'@' -f2 | cut -d':' -f1) -p $(echo $DATABASE_URL | sed 's/.*:\([0-9]*\)\/.*/\1/'); do
    echo "  PostgreSQL is unavailable - sleeping"
    sleep 2
done
echo "✅ PostgreSQL is ready"

# Wait for Redis
echo "🔴 Waiting for Redis..."
redis_host=$(echo $REDIS_URL | sed 's|redis://||' | cut -d':' -f1)
redis_port=$(echo $REDIS_URL | sed 's|redis://||' | cut -d':' -f2 | cut -d'/' -f1)
while ! nc -z $redis_host $redis_port; do
    echo "  Redis is unavailable - sleeping"
    sleep 2
done
echo "✅ Redis is ready"

# Database migrations
echo "🗄️  Running database migrations..."
python -c "
import sys
sys.path.append('/app/src')
try:
    from protein_diffusion.database.connection import run_migrations
    run_migrations()
    print('✅ Database migrations completed')
except Exception as e:
    print(f'❌ Migration failed: {e}')
    sys.exit(1)
" || {
    echo "⚠️  Migration failed, continuing with startup..."
}

# Cache warmup
echo "🔥 Warming up caches..."
python -c "
import sys
sys.path.append('/app/src')
try:
    from protein_diffusion.cache import warm_up_cache
    warm_up_cache()
    print('✅ Cache warmup completed')
except Exception as e:
    print(f'⚠️  Cache warmup failed: {e}')
" || {
    echo "⚠️  Cache warmup failed, continuing..."
}

# Model loading
echo "🤖 Loading AI models..."
python -c "
import sys
sys.path.append('/app/src')
try:
    from protein_diffusion import ProteinDiffuser
    diffuser = ProteinDiffuser()
    print('✅ AI models loaded successfully')
except Exception as e:
    print(f'⚠️  Model loading failed: {e}')
" || {
    echo "⚠️  Model loading failed, continuing..."
}

# GPU validation
if [ "$GPU_ENABLED" = "true" ]; then
    echo "🎮 Validating GPU availability..."
    python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.device_count()} devices')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  GPU not available, running in CPU mode')
" || {
    echo "⚠️  GPU validation failed"
}
fi

# Performance optimization
echo "⚡ Applying performance optimizations..."
export PYTHONOPTIMIZE=1
export OMP_NUM_THREADS=${MAX_WORKERS:-4}

# Security hardening
echo "🔒 Applying security configurations..."
umask 0077

# Health check setup
echo "💊 Setting up health checks..."
python -c "
import sys
sys.path.append('/app/src')
try:
    from protein_diffusion.health_checks import setup_health_checks
    setup_health_checks()
    print('✅ Health checks configured')
except Exception as e:
    print(f'⚠️  Health check setup failed: {e}')
" || {
    echo "⚠️  Health check setup failed"
}

# Monitoring setup
if [ "$MONITORING_ENABLED" = "true" ]; then
    echo "📊 Enabling monitoring..."
    export ENABLE_METRICS=true
    export ENABLE_TRACING=true
fi

# Determine startup mode
STARTUP_MODE=${1:-api}

case $STARTUP_MODE in
    "api"|"")
        echo "🚀 Starting API server..."
        exec gunicorn \
            --bind 0.0.0.0:8000 \
            --workers ${WORKERS:-4} \
            --worker-class ${WORKER_CLASS:-uvicorn.workers.UvicornWorker} \
            --max-requests ${MAX_REQUESTS:-1000} \
            --max-requests-jitter ${MAX_REQUESTS_JITTER:-100} \
            --timeout ${TIMEOUT:-120} \
            --keepalive ${KEEPALIVE:-5} \
            --access-logfile /app/logs/access.log \
            --error-logfile /app/logs/error.log \
            --log-level ${LOG_LEVEL:-info} \
            --preload \
            --enable-stdio-inheritance \
            protein_diffusion.main:app
        ;;
    "streaming")
        echo "🌊 Starting streaming API server..."
        exec python -m protein_diffusion.realtime_streaming_api
        ;;
    "analytics")
        echo "📈 Starting analytics dashboard..."
        exec streamlit run app.py \
            --server.port 8501 \
            --server.address 0.0.0.0 \
            --server.headless true \
            --server.enableCORS false \
            --server.enableXsrfProtection true \
            --server.maxUploadSize 200
        ;;
    "worker")
        echo "👷 Starting Celery worker..."
        exec celery -A protein_diffusion.celery worker \
            --loglevel=${LOG_LEVEL:-info} \
            --concurrency=${CELERY_WORKERS:-8} \
            --max-tasks-per-child=1000 \
            --time-limit=3600 \
            --soft-time-limit=3300
        ;;
    "beat")
        echo "⏰ Starting Celery beat scheduler..."
        exec celery -A protein_diffusion.celery beat \
            --loglevel=${LOG_LEVEL:-info} \
            --pidfile=/tmp/celerybeat.pid
        ;;
    "orchestrator")
        echo "🎭 Starting orchestration engine..."
        exec python -m protein_diffusion.nextgen_orchestration
        ;;
    "quality-gates")
        echo "🏗️  Running quality gates..."
        exec python -m protein_diffusion.nextgen_quality_gates
        ;;
    "performance-optimizer")
        echo "⚡ Starting performance optimizer..."
        exec python -m protein_diffusion.nextgen_performance_optimizer
        ;;
    "migrate")
        echo "🗄️  Running database migrations only..."
        python -c "
import sys
sys.path.append('/app/src')
from protein_diffusion.database.connection import run_migrations
run_migrations()
print('✅ Migrations completed')
"
        ;;
    "shell")
        echo "🐚 Starting interactive shell..."
        exec python -c "
import sys
sys.path.append('/app/src')
from protein_diffusion import *
print('🧬 Protein Diffusion Design Lab Shell')
print('All modules imported and ready!')
import IPython; IPython.embed()
"
        ;;
    "test")
        echo "🧪 Running tests..."
        exec python -m pytest tests/ -v \
            --cov=src/protein_diffusion \
            --cov-report=html \
            --cov-report=term \
            --tb=short
        ;;
    *)
        echo "❌ Unknown startup mode: $STARTUP_MODE"
        echo "Available modes: api, streaming, analytics, worker, beat, orchestrator, quality-gates, performance-optimizer, migrate, shell, test"
        exit 1
        ;;
esac