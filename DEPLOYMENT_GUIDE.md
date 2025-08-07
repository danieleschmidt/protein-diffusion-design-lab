# Production Deployment Guide
## Protein Diffusion Design Lab

This guide provides comprehensive instructions for deploying the Protein Diffusion Design Lab to production environments, including both Docker-based and Kubernetes deployments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Security Configuration](#security-configuration)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Backup and Recovery](#backup-and-recovery)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 8 cores (16 recommended)
- RAM: 32GB (64GB recommended)
- Storage: 200GB SSD (500GB recommended)
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3080/A100 recommended)

**Software Requirements:**
- Ubuntu 20.04+ or RHEL 8+
- Docker CE 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- PostgreSQL 15+ client tools
- SSL certificates (Let's Encrypt or commercial)

### Network Requirements

**Required Ports:**
- 80/tcp - HTTP (redirects to HTTPS)
- 443/tcp - HTTPS
- 22/tcp - SSH (secure access)
- 9090/tcp - Prometheus (monitoring)
- 3000/tcp - Grafana (dashboards)

**Optional Ports (for debugging):**
- 8000/tcp - API direct access
- 8501/tcp - Streamlit direct access
- 5432/tcp - PostgreSQL (secure networks only)
- 6379/tcp - Redis (secure networks only)

## Environment Setup

### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    curl \
    wget \
    git \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release \
    ufw

# Configure firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

### 2. Docker Installation

```bash
# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Install NVIDIA Docker for GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# Add user to docker group
sudo usermod -aG docker $USER
```

### 3. Environment Variables

Create a `.env` file in the deployment directory:

```bash
# Database Configuration
POSTGRES_PASSWORD=your_secure_database_password_here
POSTGRES_USER=protein_user
POSTGRES_DB=protein_db

# Security
JWT_SECRET=your_jwt_secret_key_here
GRAFANA_PASSWORD=your_grafana_admin_password_here

# Domain Configuration
DOMAIN_NAME=your-domain.com
SSL_EMAIL=admin@your-domain.com

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKERS=4
MAX_REQUESTS=1000
MODEL_CACHE_SIZE=5000

# Monitoring
ENABLE_MONITORING=true
METRICS_PORT=9090
DASHBOARD_PORT=3000
```

## Docker Deployment

### 1. Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/protein-diffusion-design-lab.git
cd protein-diffusion-design-lab

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run the deployment script
sudo chmod +x deployment/scripts/deploy_production.sh
sudo ./deployment/scripts/deploy_production.sh
```

### 2. Manual Deployment

```bash
# Build and start services
cd deployment
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Check service health
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs -f
```

### 3. Service Verification

```bash
# Check API health
curl -f http://localhost:8000/health

# Check web interface
curl -f http://localhost:8501/healthz

# Check database
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U protein_user -d protein_db

# Check Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
```

## Kubernetes Deployment

### 1. Cluster Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm (for additional components)
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt update && sudo apt install helm
```

### 2. Deploy to Kubernetes

```bash
# Create namespace and secrets
kubectl apply -f deployment/kubernetes/production/protein-diffusion-deployment.yaml

# Update secrets with real values
kubectl create secret generic protein-diffusion-secrets \
  --from-literal=POSTGRES_PASSWORD='your_password_here' \
  --from-literal=GRAFANA_PASSWORD='your_grafana_password_here' \
  --from-literal=JWT_SECRET='your_jwt_secret_here' \
  -n protein-diffusion --dry-run=client -o yaml | kubectl apply -f -

# Deploy the application
kubectl apply -f deployment/kubernetes/production/

# Check deployment status
kubectl get pods -n protein-diffusion
kubectl get services -n protein-diffusion
kubectl get ingress -n protein-diffusion
```

### 3. Configure Ingress

```bash
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install ingress-nginx ingress-nginx/ingress-nginx

# Install cert-manager for SSL
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.13.0 \
  --set installCRDs=true
```

## Security Configuration

### 1. SSL/TLS Setup

**Let's Encrypt (Recommended for most deployments):**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificates
sudo certbot --nginx -d your-domain.com

# Set up auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

**Manual Certificate Installation:**

```bash
# Place your certificates
sudo mkdir -p /etc/nginx/ssl
sudo cp your-cert.crt /etc/nginx/ssl/
sudo cp your-private.key /etc/nginx/ssl/
sudo chmod 600 /etc/nginx/ssl/*
```

### 2. Security Hardening

```bash
# Configure fail2ban
sudo apt install fail2ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# Edit jail.local to configure protection
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Set up log monitoring
sudo apt install logwatch
echo "0 1 * * * /usr/sbin/logwatch --output mail --mailto admin@your-domain.com --detail high" | sudo crontab -
```

### 3. Network Security

```bash
# Configure iptables (if not using ufw)
sudo iptables -A INPUT -i lo -j ACCEPT
sudo iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -j DROP

# Save iptables rules
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

## Monitoring and Alerting

### 1. Access Monitoring Dashboards

- **Grafana**: https://your-domain.com:3000
  - Username: admin
  - Password: (set in environment variables)

- **Prometheus**: https://your-domain.com:9090
  - Direct metrics access

### 2. Configure Alerting

```bash
# Edit alert rules
vim monitoring/alert_rules_production.yml

# Restart Prometheus to reload rules
docker-compose -f deployment/docker-compose.prod.yml restart prometheus
```

### 3. Set Up Notification Channels

**Slack Integration:**
```yaml
# Add to prometheus/alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'Protein Diffusion Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

**Email Notifications:**
```yaml
# Add to prometheus/alertmanager.yml
receivers:
- name: 'email'
  email_configs:
  - to: 'admin@your-domain.com'
    from: 'alerts@your-domain.com'
    smarthost: 'smtp.your-domain.com:587'
    auth_username: 'alerts@your-domain.com'
    auth_password: 'your-email-password'
    subject: 'Protein Diffusion Alert: {{ .GroupLabels.alertname }}'
```

## Backup and Recovery

### 1. Automated Backups

```bash
# The deployment script sets up automated backups
# Manual backup can be triggered:
sudo /usr/local/bin/protein-diffusion-backup.sh

# Backups are stored in:
ls -la /opt/protein-diffusion/backups/
```

### 2. Recovery Procedures

**Database Recovery:**
```bash
# Stop services
docker-compose -f deployment/docker-compose.prod.yml down

# Start only database
docker-compose -f deployment/docker-compose.prod.yml up -d postgres

# Restore from backup
docker exec -i protein-postgres psql -U protein_user -d protein_db < /path/to/backup/database.sql

# Restart all services
docker-compose -f deployment/docker-compose.prod.yml up -d
```

**Full System Recovery:**
```bash
# Restore from latest backup
LATEST_BACKUP=$(find /opt/protein-diffusion/backups -type d -name "20*" | sort -r | head -1)

# Restore application data
tar -xzf "$LATEST_BACKUP/app_data.tar.gz" -C /

# Follow database recovery procedure above
```

## Troubleshooting

### Common Issues

**1. Container Won't Start**
```bash
# Check logs
docker-compose -f deployment/docker-compose.prod.yml logs service-name

# Check resource usage
docker stats
```

**2. Database Connection Issues**
```bash
# Check database status
docker-compose -f deployment/docker-compose.prod.yml exec postgres pg_isready

# Check connection from API
docker-compose -f deployment/docker-compose.prod.yml exec protein-diffusion-api python -c "
from src.protein_diffusion.database import get_database_connection
print(get_database_connection().test_connection())
"
```

**3. GPU Not Available**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check container GPU access
docker-compose -f deployment/docker-compose.prod.yml exec protein-diffusion-api nvidia-smi
```

**4. High Memory Usage**
```bash
# Check memory usage by container
docker stats --no-stream

# Clear cached models (if safe)
docker-compose -f deployment/docker-compose.prod.yml exec protein-diffusion-api python -c "
import torch
torch.cuda.empty_cache()
"
```

### Performance Tuning

**1. Database Optimization**
```sql
-- Connect to PostgreSQL and run optimization
\c protein_db;

-- Analyze tables
ANALYZE;

-- Update statistics
UPDATE pg_stat_user_tables SET n_tup_upd = n_tup_upd + 1;

-- Check slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
```

**2. Redis Optimization**
```bash
# Connect to Redis and check stats
docker-compose -f deployment/docker-compose.prod.yml exec redis redis-cli

# Check memory usage
INFO memory

# Optimize configuration
CONFIG SET maxmemory-policy allkeys-lru
CONFIG SET maxmemory 4gb
```

## Maintenance

### Regular Tasks

**Daily:**
- Check service health
- Review error logs
- Monitor disk space
- Verify backups

**Weekly:**
- Update security patches
- Review performance metrics
- Clean up old logs
- Test backup restoration

**Monthly:**
- Review and rotate SSL certificates
- Update Docker images
- Performance optimization
- Security audit

### Update Procedures

**Application Updates:**
```bash
# Pull latest code
git pull origin main

# Rebuild and deploy
docker-compose -f deployment/docker-compose.prod.yml build --no-cache
docker-compose -f deployment/docker-compose.prod.yml up -d

# Verify deployment
./scripts/health_check.sh
```

**System Updates:**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker
sudo apt install docker-ce docker-ce-cli containerd.io

# Restart services if needed
docker-compose -f deployment/docker-compose.prod.yml restart
```

### Scaling

**Horizontal Scaling:**
```bash
# Scale API containers
docker-compose -f deployment/docker-compose.prod.yml up -d --scale protein-diffusion-api=3

# Scale web interface
docker-compose -f deployment/docker-compose.prod.yml up -d --scale protein-diffusion-web=2
```

**Vertical Scaling:**
```bash
# Edit docker-compose.prod.yml to increase resource limits
# Restart services
docker-compose -f deployment/docker-compose.prod.yml down
docker-compose -f deployment/docker-compose.prod.yml up -d
```

### Support and Documentation

- **Technical Documentation**: `/docs/`
- **API Documentation**: `https://your-domain.com:8000/docs`
- **Monitoring Dashboards**: `https://your-domain.com:3000`
- **Health Endpoints**: 
  - API: `https://your-domain.com:8000/health`
  - Web: `https://your-domain.com:8501/healthz`

For additional support, consult the project repository or contact the development team.

---

## Security Considerations

⚠️ **Important Security Notes:**

1. **Change Default Passwords**: All default passwords must be changed before production deployment
2. **Network Security**: Use firewalls and VPNs to restrict access to sensitive ports
3. **SSL/TLS**: Always use HTTPS in production with valid certificates
4. **Regular Updates**: Keep all components updated with security patches
5. **Access Logging**: Enable and monitor access logs for security events
6. **Backup Security**: Ensure backups are encrypted and stored securely
7. **API Keys**: Use strong API keys and rotate them regularly
8. **Database Security**: Use strong passwords and limit database access
9. **Container Security**: Regularly scan container images for vulnerabilities
10. **Monitoring**: Set up alerts for security events and anomalies