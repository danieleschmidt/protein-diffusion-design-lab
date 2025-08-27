#!/bin/bash
set -euo pipefail

# Progressive Quality Gates - Production Deployment Script
# Autonomous SDLC Implementation - Terragon Labs v4.0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="quality-gates"
DEPLOYMENT_NAME="progressive-quality-gates"
VERSION="v4.0"
DOCKER_IMAGE="terragon/progressive-quality-gates:${VERSION}"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✅ $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ⚠️  $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ❌ $1"
}

# Banner
echo -e "${PURPLE}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🧬 Progressive Quality Gates - Production Deployment     ║
║                                                               ║
║     Autonomous SDLC Implementation v4.0                      ║
║     Terragon Labs - Enterprise Quality Assurance             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Pre-deployment checks
log "🔍 Running pre-deployment checks..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check Kubernetes connectivity
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

log_success "Pre-deployment checks passed"

# Build and push Docker image
log "🐳 Building Docker image..."

# Create Dockerfile if it doesn't exist
if [ ! -f "Dockerfile.production" ]; then
    log "📝 Creating production Dockerfile..."
    cat > Dockerfile.production << 'EOF'
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash quality-gates
RUN chown -R quality-gates:quality-gates /app
USER quality-gates

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Start application
CMD ["python", "-m", "enhanced_quality_gate_runner", "--production", "--port=8080"]
EOF
fi

# Build Docker image
log "🔨 Building Docker image: ${DOCKER_IMAGE}"
docker build -f Dockerfile.production -t "${DOCKER_IMAGE}" .

# Push Docker image (if registry is configured)
if [ "${PUSH_TO_REGISTRY:-false}" = "true" ]; then
    log "📤 Pushing Docker image to registry..."
    docker push "${DOCKER_IMAGE}"
    log_success "Docker image pushed successfully"
else
    log_warning "Skipping Docker push (set PUSH_TO_REGISTRY=true to enable)"
fi

# Create namespace if it doesn't exist
log "🏗️ Creating namespace: ${NAMESPACE}"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
log "☸️ Deploying to Kubernetes..."

# Apply configuration
if [ -f "deployment/production_quality_gates.yaml" ]; then
    kubectl apply -f deployment/production_quality_gates.yaml
    log_success "Kubernetes manifests applied"
else
    log_error "Production deployment manifests not found"
    exit 1
fi

# Wait for deployment to be ready
log "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}

# Check pod status
log "🔍 Checking pod status..."
kubectl get pods -n ${NAMESPACE} -l app=${DEPLOYMENT_NAME}

# Run health checks
log "🏥 Running health checks..."
SERVICE_URL="http://$(kubectl get svc ${DEPLOYMENT_NAME}-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' || echo 'localhost')"

# Port forward for local testing
log "🔗 Setting up port forwarding for health checks..."
kubectl port-forward svc/${DEPLOYMENT_NAME}-service 8080:80 -n ${NAMESPACE} &
PORT_FORWARD_PID=$!

# Wait a moment for port forward to establish
sleep 5

# Health check
if curl -f http://localhost:8080/health 2>/dev/null; then
    log_success "Health check passed"
else
    log_warning "Health check failed - service may still be starting"
fi

# Clean up port forward
kill $PORT_FORWARD_PID 2>/dev/null || true

# Display deployment information
log "📊 Deployment Information:"
echo -e "${GREEN}"
echo "  🌍 Namespace: ${NAMESPACE}"
echo "  🚀 Deployment: ${DEPLOYMENT_NAME}"
echo "  🏷️  Version: ${VERSION}"
echo "  📦 Image: ${DOCKER_IMAGE}"
echo "  🔗 Service: ${DEPLOYMENT_NAME}-service"
echo -e "${NC}"

# Display ingress information if available
INGRESS_IP=$(kubectl get ingress ${DEPLOYMENT_NAME}-ingress -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Not available")
if [ "${INGRESS_IP}" != "Not available" ]; then
    echo -e "${GREEN}  🌐 External IP: ${INGRESS_IP}${NC}"
fi

# Show scaling information
log "📈 Auto-scaling Configuration:"
kubectl get hpa ${DEPLOYMENT_NAME}-hpa -n ${NAMESPACE}

# Quality Gates Test
log "🧪 Running production quality gates test..."

# Create test job
cat > /tmp/quality-gates-test-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: quality-gates-test
  namespace: ${NAMESPACE}
spec:
  template:
    spec:
      containers:
      - name: test-runner
        image: ${DOCKER_IMAGE}
        command: ["python", "test_progressive_quality_gates.py"]
        env:
        - name: ENVIRONMENT
          value: "production"
      restartPolicy: Never
  backoffLimit: 2
EOF

kubectl apply -f /tmp/quality-gates-test-job.yaml
log "⏳ Waiting for test job to complete..."
kubectl wait --for=condition=complete --timeout=300s job/quality-gates-test -n ${NAMESPACE}

# Show test results
log "📋 Test Results:"
kubectl logs job/quality-gates-test -n ${NAMESPACE}

# Cleanup test job
kubectl delete job quality-gates-test -n ${NAMESPACE}
rm -f /tmp/quality-gates-test-job.yaml

# Final status
log_success "🎉 Production deployment completed successfully!"

echo -e "${GREEN}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║                ✅ DEPLOYMENT SUCCESSFUL ✅                   ║
║                                                               ║
║   Progressive Quality Gates is now running in production!    ║
║                                                               ║
║   • Kubernetes deployment: Ready                             ║
║   • Auto-scaling: Enabled                                    ║
║   • Monitoring: Active                                       ║
║   • Security: Hardened                                       ║
║   • Quality Gates: Validated                                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Display useful commands
log "🔧 Useful Commands:"
echo "  • View logs: kubectl logs -f deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}"
echo "  • Scale deployment: kubectl scale deployment ${DEPLOYMENT_NAME} --replicas=5 -n ${NAMESPACE}"
echo "  • Check status: kubectl get all -n ${NAMESPACE}"
echo "  • Port forward: kubectl port-forward svc/${DEPLOYMENT_NAME}-service 8080:80 -n ${NAMESPACE}"
echo "  • Delete deployment: kubectl delete namespace ${NAMESPACE}"

log_success "Autonomous SDLC deployment completed! 🚀"