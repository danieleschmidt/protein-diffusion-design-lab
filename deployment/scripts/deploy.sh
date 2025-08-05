#!/bin/bash
set -e

# Protein Diffusion Design Lab - Production Deployment Script
# This script deploys the application to production environment

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
BUILD_TAG=${BUILD_TAG:-latest}
PROJECT_NAME="protein-diffusion"
COMPOSE_FILE="deployment/docker-compose.prod.yml"
ENV_FILE="deployment/.env.prod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Environment file $ENV_FILE not found"
        log_info "Please create the environment file with required variables"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Load environment variables
load_env() {
    log_info "Loading environment variables from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
    
    # Validate required environment variables
    required_vars=(
        "POSTGRES_PASSWORD"
        "GRAFANA_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log_success "Environment variables loaded"
}

# Backup current deployment
backup_current() {
    log_info "Creating backup of current deployment..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        log_info "Backing up database..."
        docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U protein_user protein_db > "$BACKUP_DIR/database.sql"
        log_success "Database backup created"
    fi
    
    # Backup volumes
    log_info "Backing up volumes..."
    docker run --rm \
        -v protein-diffusion_protein_models:/backup-source:ro \
        -v "$(pwd)/$BACKUP_DIR":/backup-dest \
        alpine tar czf /backup-dest/models.tar.gz -C /backup-source .
    
    log_success "Backup created in $BACKUP_DIR"
}

# Build images
build_images() {
    log_info "Building Docker images..."
    
    # Build main application image
    docker build -t "${PROJECT_NAME}:${BUILD_TAG}" -f Dockerfile .
    
    # Build web interface image
    docker build -t "${PROJECT_NAME}-web:${BUILD_TAG}" -f Dockerfile --target web .
    
    log_success "Images built successfully"
}

# Deploy application
deploy() {
    log_info "Deploying application..."
    
    # Create necessary directories
    mkdir -p deployment/nginx/logs
    mkdir -p logs
    
    # Deploy with Docker Compose
    docker-compose -f "$COMPOSE_FILE" up -d --remove-orphans
    
    log_success "Application deployed"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."
    
    # Services to check
    services=(
        "protein-redis:6379"
        "protein-postgres:5432"
        "protein-diffusion-api:8000"
        "protein-diffusion-web:8501"
    )
    
    for service in "${services[@]}"; do
        service_name=$(echo "$service" | cut -d: -f1)
        service_port=$(echo "$service" | cut -d: -f2)
        
        log_info "Waiting for $service_name to be ready..."
        
        timeout=300  # 5 minutes
        counter=0
        
        while ! docker-compose -f "$COMPOSE_FILE" exec "$service_name" sh -c "nc -z localhost $service_port" &> /dev/null; do
            if [[ $counter -ge $timeout ]]; then
                log_error "Timeout waiting for $service_name"
                return 1
            fi
            sleep 5
            counter=$((counter + 5))
        done
        
        log_success "$service_name is ready"
    done
    
    # Wait for health checks
    log_info "Waiting for health checks to pass..."
    sleep 30
    
    # Check container health
    unhealthy_containers=$(docker-compose -f "$COMPOSE_FILE" ps --filter "health=unhealthy" -q)
    if [[ -n "$unhealthy_containers" ]]; then
        log_error "Some containers are unhealthy:"
        docker-compose -f "$COMPOSE_FILE" ps --filter "health=unhealthy"
        return 1
    fi
    
    log_success "All services are healthy"
}

# Run post-deployment tests
run_tests() {
    log_info "Running post-deployment tests..."
    
    # Test API health endpoint
    if curl -f http://localhost/api/health &> /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Test web interface
    if curl -f http://localhost:8501/healthz &> /dev/null; then
        log_success "Web interface health check passed"
    else
        log_error "Web interface health check failed"
        return 1
    fi
    
    # Test database connection
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U protein_user -d protein_db &> /dev/null; then
        log_success "Database connection test passed"
    else
        log_error "Database connection test failed"
        return 1
    fi
    
    # Test Redis connection
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis connection test passed"
    else
        log_error "Redis connection test failed"
        return 1
    fi
    
    log_success "All post-deployment tests passed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo
    
    # Show running containers
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    
    # Show service URLs
    log_info "Service URLs:"
    echo "  API: http://localhost/api/"
    echo "  Web Interface: http://localhost:8501/"
    echo "  Grafana: http://localhost:3000/"
    echo "  Prometheus: http://localhost:9090/"
    echo
    
    # Show resource usage
    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
        $(docker-compose -f "$COMPOSE_FILE" ps -q)
}

# Cleanup old images and containers
cleanup() {
    log_info "Cleaning up old images and containers..."
    
    # Remove old images
    docker image prune -f
    
    # Remove unused volumes (be careful with this)
    # docker volume prune -f
    
    log_success "Cleanup completed"
}

# Rollback function
rollback() {
    log_warn "Rolling back deployment..."
    
    # Stop current deployment
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore from latest backup
    LATEST_BACKUP=$(find backups -name "*" -type d | sort | tail -n 1)
    
    if [[ -n "$LATEST_BACKUP" && -d "$LATEST_BACKUP" ]]; then
        log_info "Restoring from backup: $LATEST_BACKUP"
        
        # Restore database
        if [[ -f "$LATEST_BACKUP/database.sql" ]]; then
            docker-compose -f "$COMPOSE_FILE" up -d postgres
            sleep 10
            docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U protein_user -d protein_db < "$LATEST_BACKUP/database.sql"
        fi
        
        # Restore volumes
        if [[ -f "$LATEST_BACKUP/models.tar.gz" ]]; then
            docker run --rm \
                -v protein-diffusion_protein_models:/backup-dest \
                -v "$(pwd)/$LATEST_BACKUP":/backup-source:ro \
                alpine tar xzf /backup-source/models.tar.gz -C /backup-dest
        fi
        
        log_success "Rollback completed"
    else
        log_error "No backup found for rollback"
        exit 1
    fi
}

# Main deployment function
main() {
    log_info "Starting deployment of Protein Diffusion Design Lab"
    log_info "Environment: $ENVIRONMENT"
    log_info "Build Tag: $BUILD_TAG"
    echo
    
    # Check if rollback is requested
    if [[ "$1" == "rollback" ]]; then
        rollback
        exit 0
    fi
    
    # Pre-deployment checks
    check_root
    check_prerequisites
    load_env
    
    # Create backup
    backup_current
    
    # Build and deploy
    build_images
    deploy
    
    # Post-deployment verification
    wait_for_services
    run_tests
    
    # Show status
    show_status
    
    # Cleanup
    cleanup
    
    log_success "Deployment completed successfully!"
    log_info "Monitor the services and check logs for any issues"
}

# Error handling
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Parse command line arguments
case "$1" in
    "rollback")
        main rollback
        ;;
    "status")
        show_status
        ;;
    *)
        main
        ;;
esac