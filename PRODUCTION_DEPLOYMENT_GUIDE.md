# Production Deployment Guide
## Protein Diffusion Design Lab - Enterprise Edition

This guide provides comprehensive instructions for deploying the enhanced Protein Diffusion Design Lab platform to production environments.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or compatible container environment
- **CPU**: Minimum 8 cores, 16+ cores recommended
- **Memory**: Minimum 32GB RAM, 64GB+ recommended
- **Storage**: Minimum 500GB SSD, 1TB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Software Dependencies
- **Python**: 3.9 or higher
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.21+ (for orchestrated deployment)
- **kubectl**: Compatible with your Kubernetes version
- **Helm**: 3.0+ (recommended for Kubernetes deployments)

## 🚀 Deployment Options

### Option 1: Standalone Python Deployment

#### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd protein-diffusion-design-lab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configuration
```bash
# Copy example configuration
cp config/production.example.yml config/production.yml

# Edit configuration file
nano config/production.yml
```

#### 3. Database Setup
```bash
# Initialize database
python scripts/init_database.py

# Run migrations
python scripts/migrate.py
```

#### 4. Start Services
```bash
# Start the application
python -m src.protein_diffusion.app --config config/production.yml
```

### Option 2: Docker Deployment

#### 1. Build Container
```bash
# Build production image
docker build -t protein-diffusion-lab:latest .

# Or use pre-built image
docker pull protein-diffusion-lab:latest
```

#### 2. Run Container
```bash
docker run -d \
  --name protein-diffusion-lab \
  --restart unless-stopped \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  -e ENVIRONMENT=production \
  protein-diffusion-lab:latest
```

#### 3. With Docker Compose
```yaml
version: '3.8'
services:
  protein-diffusion-lab:
    image: protein-diffusion-lab:latest
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./cache:/app/cache
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:pass@db:5432/protein_db
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: protein_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Option 3: Kubernetes Deployment (Recommended)

#### 1. Prepare Kubernetes Manifests
```bash
# Use the built-in deployment manager
python -c "
from src.protein_diffusion.deployment_manager import DeploymentManager
dm = DeploymentManager()
dm.deploy_complete_stack()
"
```

#### 2. Apply Manifests
```bash
# Create namespace
kubectl create namespace protein-diffusion

# Apply all manifests
kubectl apply -f k8s/ -n protein-diffusion
```

#### 3. Verify Deployment
```bash
# Check pod status
kubectl get pods -n protein-diffusion

# Check services
kubectl get services -n protein-diffusion

# Check ingress
kubectl get ingress -n protein-diffusion
```

## ⚙️ Configuration

### Environment Variables
```bash
# Application settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database configuration
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379

# Performance settings
MAX_WORKERS=16
CACHE_SIZE=1000
GPU_ENABLED=true

# Security settings
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret
ALLOWED_HOSTS=your-domain.com

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

### Configuration Files

#### production.yml
```yaml
server:
  host: 0.0.0.0
  port: 8080
  workers: 16

database:
  url: postgresql://user:pass@localhost:5432/protein_db
  pool_size: 20
  max_overflow: 30

cache:
  redis_url: redis://localhost:6379
  default_ttl: 3600
  max_memory: 2GB

performance:
  enable_gpu: true
  batch_size: 32
  max_sequence_length: 512
  
monitoring:
  enabled: true
  prometheus_port: 9090
  log_level: INFO

security:
  secret_key: ${SECRET_KEY}
  jwt_secret: ${JWT_SECRET}
  cors_origins:
    - https://your-domain.com
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Generate certificates (if not using external CA)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure nginx reverse proxy
cat > /etc/nginx/sites-available/protein-diffusion << EOF
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
```

### Firewall Configuration
```bash
# Ubuntu/Debian with ufw
ufw allow 22/tcp      # SSH
ufw allow 80/tcp      # HTTP
ufw allow 443/tcp     # HTTPS
ufw allow 9090/tcp    # Metrics (internal only)
ufw enable
```

## 📊 Monitoring Setup

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'protein-diffusion'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
    metrics_path: /metrics
```

### Grafana Dashboard
```bash
# Import pre-built dashboard
curl -X POST \
  http://admin:admin@grafana:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana-dashboard.json
```

### Log Aggregation
```yaml
# fluent-bit.conf
[SERVICE]
    Flush         1
    Log_Level     info

[INPUT]
    Name              tail
    Path              /app/logs/*.log
    Parser            json
    Tag               protein-diffusion.*

[OUTPUT]
    Name              es
    Match             *
    Host              elasticsearch
    Port              9200
    Index             protein-diffusion-logs
```

## 🔧 Performance Tuning

### Database Optimization
```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### Cache Configuration
```python
# Redis tuning
# In redis.conf
maxmemory 8gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### GPU Optimization
```bash
# NVIDIA settings
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -pl 250  # Set power limit (adjust based on your GPU)

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use multiple GPUs
export CUDA_CACHE_PATH=/tmp/cuda_cache
```

## 🚨 Health Checks

### Application Health
```bash
# Health check endpoint
curl -f http://localhost:8080/health || exit 1

# Detailed health check
curl http://localhost:8080/health/detailed
```

### Kubernetes Health Checks
```yaml
# In deployment manifest
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

## 📁 Backup Strategy

### Database Backup
```bash
#!/bin/bash
# backup-db.sh
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump protein_db > /backups/protein_db_$DATE.sql
find /backups -name "protein_db_*.sql" -mtime +7 -delete
```

### Application Data Backup
```bash
#!/bin/bash
# backup-data.sh
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf /backups/protein_data_$DATE.tar.gz /app/data /app/cache
find /backups -name "protein_data_*.tar.gz" -mtime +30 -delete
```

### Automated Backup
```bash
# Add to crontab
0 2 * * * /scripts/backup-db.sh
0 3 * * * /scripts/backup-data.sh
```

## 🔄 Deployment Automation

### CI/CD Pipeline (GitHub Actions)
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build and push Docker image
        run: |
          docker build -t protein-diffusion:${{ github.sha }} .
          docker push protein-diffusion:${{ github.sha }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/protein-diffusion \
            protein-diffusion=protein-diffusion:${{ github.sha }}
          kubectl rollout status deployment/protein-diffusion
```

### Blue-Green Deployment
```bash
#!/bin/bash
# blue-green-deploy.sh
NEW_VERSION=$1
CURRENT_COLOR=$(kubectl get service protein-diffusion -o jsonpath='{.spec.selector.version}')

if [ "$CURRENT_COLOR" = "blue" ]; then
    NEW_COLOR="green"
else
    NEW_COLOR="blue"
fi

# Deploy new version
kubectl set image deployment/protein-diffusion-$NEW_COLOR \
    protein-diffusion=protein-diffusion:$NEW_VERSION

# Wait for rollout
kubectl rollout status deployment/protein-diffusion-$NEW_COLOR

# Switch traffic
kubectl patch service protein-diffusion -p '{"spec":{"selector":{"version":"'$NEW_COLOR'"}}}'

echo "Deployment complete. Traffic switched to $NEW_COLOR"
```

## 🆘 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -n protein-diffusion

# Scale down if needed
kubectl scale deployment protein-diffusion --replicas=2
```

#### Slow Generation Times
```bash
# Check GPU utilization
nvidia-smi

# Check cache hit rates
curl http://localhost:8080/metrics | grep cache_hit_rate

# Enable performance profiling
export ENABLE_PROFILING=true
```

#### Database Connection Issues
```bash
# Check connections
kubectl logs deployment/protein-diffusion | grep database

# Test connection
psql postgresql://user:pass@host:5432/db -c "SELECT 1"
```

### Log Analysis
```bash
# Application logs
kubectl logs -f deployment/protein-diffusion

# Error logs only
kubectl logs deployment/protein-diffusion | grep ERROR

# Specific timeframe
kubectl logs deployment/protein-diffusion --since=1h
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks
- **Daily**: Check application health and error logs
- **Weekly**: Review performance metrics and capacity usage
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Performance tuning and optimization review

### Emergency Contacts
- **DevOps Team**: devops@company.com
- **Platform Team**: platform@company.com
- **On-call**: +1-555-0123

### Useful Commands
```bash
# Quick health check
kubectl get all -n protein-diffusion

# View recent events
kubectl get events -n protein-diffusion --sort-by='.lastTimestamp'

# Emergency scale down
kubectl scale deployment protein-diffusion --replicas=0

# Emergency rollback
kubectl rollout undo deployment/protein-diffusion
```

## 🔗 Additional Resources

- [Application Documentation](./README.md)
- [API Documentation](./docs/api.md)
- [Architecture Guide](./docs/architecture.md)
- [Performance Tuning Guide](./docs/performance.md)
- [Security Best Practices](./docs/security.md)

---

**📞 Need Help?**
If you encounter issues during deployment, please refer to the troubleshooting section or contact the support team.

**🔄 Last Updated**: December 2024
**📋 Version**: 5.0 (Enterprise Edition)