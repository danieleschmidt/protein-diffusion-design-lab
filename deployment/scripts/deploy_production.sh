#!/bin/bash
# Production deployment script for Protein Diffusion Design Lab
# This script automates the complete deployment process with safety checks

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"
BACKUP_DIR="/opt/protein-diffusion/backups/$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="/var/log/protein-diffusion-deploy.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Configuration validation
validate_config() {
    log "Validating deployment configuration..."
    
    # Check required environment variables
    local required_vars=(
        "POSTGRES_PASSWORD"
        "GRAFANA_PASSWORD" 
        "DOMAIN_NAME"
        "SSL_EMAIL"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate configuration files
    if [ ! -f "$DEPLOYMENT_DIR/config/production.yaml" ]; then
        error "Production configuration file not found"
        exit 1
    fi
    
    # Validate Docker Compose files
    if [ ! -f "$DEPLOYMENT_DIR/docker-compose.prod.yml" ]; then
        error "Production Docker Compose file not found"
        exit 1
    fi
    
    success "Configuration validation passed"
}

# System prerequisites check
check_prerequisites() {
    log "Checking system prerequisites..."
    
    # Check if running as root for system setup
    if [ "$EUID" -ne 0 ]; then
        error "This script must be run as root for initial setup"
        exit 1
    fi
    
    # Check available disk space (minimum 50GB)
    local available_space
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 52428800 ]; then  # 50GB in KB
        error "Insufficient disk space. At least 50GB required"
        exit 1
    fi
    
    # Check available RAM (minimum 16GB)
    local available_ram
    available_ram=$(free -k | awk '/^Mem:/ {print $2}')
    if [ "$available_ram" -lt 16777216 ]; then  # 16GB in KB
        warning "Less than 16GB RAM available. Performance may be impacted"
    fi
    
    # Install required packages if missing
    local packages=("docker.io" "docker-compose" "nginx" "certbot" "python3-certbot-nginx")
    for package in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii.*$package"; then
            log "Installing $package..."
            apt-get update && apt-get install -y "$package"
        fi
    done
    
    # Check Docker
    if ! systemctl is-active --quiet docker; then
        log "Starting Docker service..."
        systemctl start docker
        systemctl enable docker
    fi
    
    success "Prerequisites check completed"
}

# Security setup
setup_security() {
    log "Setting up security configurations..."
    
    # Create dedicated user for the application
    if ! id protein_user &>/dev/null; then
        log "Creating application user..."
        useradd -r -s /bin/false -d /opt/protein-diffusion protein_user
    fi
    
    # Set up proper directory permissions
    mkdir -p /opt/protein-diffusion/{data,models,logs,backups}
    chown -R protein_user:protein_user /opt/protein-diffusion
    chmod 750 /opt/protein-diffusion
    
    # Configure firewall
    if command -v ufw >/dev/null 2>&1; then
        log "Configuring firewall..."
        ufw --force enable
        ufw default deny incoming
        ufw default allow outgoing
        ufw allow ssh
        ufw allow 80/tcp
        ufw allow 443/tcp
        ufw allow 9090/tcp  # Prometheus
        ufw allow 3000/tcp  # Grafana
    fi
    
    # Generate strong random secrets if not provided
    if [ -z "${JWT_SECRET:-}" ]; then
        export JWT_SECRET=$(openssl rand -base64 32)
        echo "JWT_SECRET=$JWT_SECRET" >> /etc/environment
    fi
    
    success "Security setup completed"
}

# SSL certificate setup
setup_ssl() {
    log "Setting up SSL certificates..."
    
    if [ -z "${DOMAIN_NAME:-}" ]; then
        warning "DOMAIN_NAME not set, skipping SSL setup"
        return
    fi
    
    # Check if certificates already exist
    if [ ! -f "/etc/letsencrypt/live/$DOMAIN_NAME/fullchain.pem" ]; then
        log "Obtaining SSL certificate for $DOMAIN_NAME..."
        
        # Stop nginx temporarily
        systemctl stop nginx || true
        
        # Obtain certificate
        certbot certonly --standalone \
            --email "$SSL_EMAIL" \
            --agree-tos \
            --no-eff-email \
            -d "$DOMAIN_NAME"
        
        # Set up auto-renewal
        (crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet") | crontab -
    else
        log "SSL certificate already exists for $DOMAIN_NAME"
    fi
    
    success "SSL setup completed"
}

# Database initialization
initialize_database() {
    log "Initializing database..."
    
    cd "$DEPLOYMENT_DIR"
    
    # Start only the database first
    docker-compose -f docker-compose.prod.yml up -d postgres redis
    
    # Wait for database to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U protein_user -d protein_db; then
            break
        fi
        
        log "Waiting for database... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        error "Database failed to start within expected time"
        exit 1
    fi
    
    # Run migrations
    log "Running database migrations..."
    docker-compose -f docker-compose.prod.yml exec -T protein-diffusion-api python -m src.protein_diffusion.database.migrations.run_migrations
    
    success "Database initialization completed"
}

# Application deployment
deploy_application() {
    log "Deploying application..."
    
    cd "$DEPLOYMENT_DIR"
    
    # Build images
    log "Building Docker images..."
    docker-compose -f docker-compose.prod.yml build --no-cache
    
    # Pull latest base images
    docker-compose -f docker-compose.prod.yml pull
    
    # Start services
    log "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    local services=("protein-diffusion-api" "protein-diffusion-web" "redis" "postgres")
    
    for service in "${services[@]}"; do
        local max_attempts=20
        local attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if docker-compose -f docker-compose.prod.yml ps "$service" | grep -q "healthy"; then
                success "$service is healthy"
                break
            fi
            
            log "Waiting for $service to be healthy... (attempt $attempt/$max_attempts)"
            sleep 15
            ((attempt++))
        done
        
        if [ $attempt -gt $max_attempts ]; then
            error "$service failed health check"
            docker-compose -f docker-compose.prod.yml logs "$service"
            exit 1
        fi
    done
    
    success "Application deployment completed"
}

# Monitoring setup
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    cd "$DEPLOYMENT_DIR"
    
    # Start monitoring services
    docker-compose -f docker-compose.prod.yml up -d prometheus grafana node-exporter
    
    # Wait for Grafana to be ready
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
            break
        fi
        
        log "Waiting for Grafana... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    # Configure Grafana dashboards (if API is available)
    if [ $attempt -le $max_attempts ]; then
        log "Configuring Grafana dashboards..."
        # Dashboard configuration would go here
        success "Monitoring setup completed"
    else
        warning "Grafana setup may need manual configuration"
    fi
}

# Backup setup
setup_backups() {
    log "Setting up backup system..."
    
    # Create backup directories
    mkdir -p "$BACKUP_DIR"
    
    # Create backup script
    cat > /usr/local/bin/protein-diffusion-backup.sh << 'EOF'
#!/bin/bash
# Automated backup script for Protein Diffusion

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/protein-diffusion/backups/$BACKUP_DATE"
mkdir -p "$BACKUP_DIR"

# Database backup
docker exec protein-postgres pg_dump -U protein_user protein_db > "$BACKUP_DIR/database.sql"

# Redis backup
docker exec protein-redis redis-cli SAVE
docker cp protein-redis:/data/dump.rdb "$BACKUP_DIR/redis.rdb"

# Application data backup
tar -czf "$BACKUP_DIR/app_data.tar.gz" /opt/protein-diffusion/data /opt/protein-diffusion/models

# Cleanup old backups (keep last 7 days)
find /opt/protein-diffusion/backups -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR"
EOF
    
    chmod +x /usr/local/bin/protein-diffusion-backup.sh
    
    # Schedule daily backups
    (crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/protein-diffusion-backup.sh") | crontab -
    
    success "Backup system configured"
}

# Health checks and validation
validate_deployment() {
    log "Validating deployment..."
    
    # Check API health
    local api_health
    api_health=$(curl -s -w "%{http_code}" http://localhost:8000/health -o /dev/null)
    if [ "$api_health" != "200" ]; then
        error "API health check failed (HTTP $api_health)"
        exit 1
    fi
    
    # Check web interface
    local web_health
    web_health=$(curl -s -w "%{http_code}" http://localhost:8501/healthz -o /dev/null)
    if [ "$web_health" != "200" ]; then
        error "Web interface health check failed (HTTP $web_health)"
        exit 1
    fi
    
    # Check database connection
    if ! docker-compose -f "$DEPLOYMENT_DIR/docker-compose.prod.yml" exec -T postgres pg_isready -U protein_user -d protein_db; then
        error "Database connection check failed"
        exit 1
    fi
    
    # Check Redis
    if ! docker-compose -f "$DEPLOYMENT_DIR/docker-compose.prod.yml" exec -T redis redis-cli ping | grep -q PONG; then
        error "Redis connection check failed"
        exit 1
    fi
    
    success "Deployment validation passed"
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    # Remove old Docker images
    docker image prune -f
    
    # Clear temporary files
    rm -rf /tmp/protein-diffusion-*
    
    success "Cleanup completed"
}

# Rollback function
rollback() {
    error "Deployment failed. Initiating rollback..."
    
    cd "$DEPLOYMENT_DIR"
    
    # Stop current deployment
    docker-compose -f docker-compose.prod.yml down
    
    # Restore from backup if available
    local latest_backup
    latest_backup=$(find /opt/protein-diffusion/backups -type d -name "20*" | sort -r | head -1)
    
    if [ -n "$latest_backup" ] && [ -d "$latest_backup" ]; then
        log "Restoring from backup: $latest_backup"
        
        # Restore database
        if [ -f "$latest_backup/database.sql" ]; then
            docker-compose -f docker-compose.prod.yml up -d postgres
            sleep 10
            docker exec -i protein-postgres psql -U protein_user -d protein_db < "$latest_backup/database.sql"
        fi
        
        # Restore Redis
        if [ -f "$latest_backup/redis.rdb" ]; then
            docker cp "$latest_backup/redis.rdb" protein-redis:/data/dump.rdb
            docker restart protein-redis
        fi
        
        # Restore application data
        if [ -f "$latest_backup/app_data.tar.gz" ]; then
            tar -xzf "$latest_backup/app_data.tar.gz" -C /
        fi
    fi
    
    error "Rollback completed. Please check logs and retry deployment."
    exit 1
}

# Main deployment function
main() {
    log "Starting production deployment of Protein Diffusion Design Lab..."
    
    # Set trap for cleanup on exit
    trap cleanup EXIT
    trap rollback ERR
    
    # Execute deployment steps
    validate_config
    check_prerequisites
    setup_security
    setup_ssl
    initialize_database
    deploy_application
    setup_monitoring
    setup_backups
    validate_deployment
    
    success "🎉 Production deployment completed successfully!"
    
    echo
    log "Access your application at:"
    echo "  • API: https://${DOMAIN_NAME:-localhost}:8000"
    echo "  • Web Interface: https://${DOMAIN_NAME:-localhost}:8501"
    echo "  • Monitoring: https://${DOMAIN_NAME:-localhost}:3000"
    echo "  • Metrics: https://${DOMAIN_NAME:-localhost}:9090"
    echo
    log "Important files:"
    echo "  • Logs: /var/log/protein-diffusion-deploy.log"
    echo "  • Data: /opt/protein-diffusion/data"
    echo "  • Backups: /opt/protein-diffusion/backups"
    echo
    log "Next steps:"
    echo "  1. Configure DNS to point to this server"
    echo "  2. Update firewall rules as needed"
    echo "  3. Set up monitoring alerts"
    echo "  4. Test the backup system"
    
    return 0
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi