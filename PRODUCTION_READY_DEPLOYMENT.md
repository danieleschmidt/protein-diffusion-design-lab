# 🚀 Production Deployment Guide - Protein Diffusion Design Lab

## 📋 Overview

This guide provides comprehensive instructions for deploying the Protein Diffusion Design Lab in production environments. The system has been enhanced through autonomous SDLC execution with enterprise-grade capabilities.

## 🏗️ Architecture Summary

The system implements a **progressive enhancement strategy** across 3 generations:

- **Generation 1 (WORK)**: Basic functionality with comprehensive CLI and testing
- **Generation 2 (ROBUST)**: Enhanced error handling, security, and monitoring  
- **Generation 3 (SCALE)**: Performance optimization and distributed scaling

## 🔧 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 10.15+
- **Python**: 3.9+
- **Memory**: 16GB RAM (32GB recommended)
- **Storage**: 50GB free space
- **GPU**: Optional but recommended (16GB+ VRAM)

### Dependencies
```bash
# Core dependencies
pip install torch>=2.0.0 torchvision>=0.15.0
pip install numpy>=1.24.0 scipy>=1.10.0 pandas>=2.0.0
pip install streamlit>=1.28.0 plotly>=5.15.0
pip install biopython>=1.81 selfies>=2.1.1

# Optional ML dependencies
pip install esm>=2.0.0 fair-esm>=2.0.0

# Monitoring dependencies  
pip install psutil>=5.9.0
```

## 📦 Installation Methods

### Method 1: Direct Installation
```bash
# Clone repository
git clone https://github.com/your-org/protein-diffusion-design-lab.git
cd protein-diffusion-design-lab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Method 2: Docker Deployment
```bash
# Build production image
docker build -f Dockerfile.production -t protein-diffusion:latest .

# Run container
docker run -d \
  --name protein-diffusion \
  -p 8501:8501 \
  -v /data:/app/data \
  --gpus all \
  protein-diffusion:latest
```

### Method 3: Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/production/
```

## ⚙️ Configuration

### Environment Variables
```bash
# Core Configuration
export PROTEIN_DIFFUSION_ENV=production
export PROTEIN_DIFFUSION_LOG_LEVEL=INFO
export PROTEIN_DIFFUSION_DATA_DIR=/data/protein-diffusion

# Performance Configuration
export PROTEIN_DIFFUSION_CACHE_SIZE=1000
export PROTEIN_DIFFUSION_BATCH_SIZE=32
export PROTEIN_DIFFUSION_MAX_WORKERS=8

# Security Configuration
export PROTEIN_DIFFUSION_API_KEY=your-secure-api-key
export PROTEIN_DIFFUSION_ENABLE_AUTH=true
export PROTEIN_DIFFUSION_RATE_LIMIT=100

# GPU Configuration (if available)
export CUDA_VISIBLE_DEVICES=0,1
export PROTEIN_DIFFUSION_GPU_MEMORY_FRACTION=0.8
```

### Configuration File
Create `config/production.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 8501
  workers: 8

model:
  device: "cuda"
  batch_size: 32
  max_length: 256
  cache_size: 1000

security:
  enable_auth: true
  rate_limit: 100
  api_key_required: true

monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30

performance:
  auto_scaling: true
  memory_limit: "8GB"
  gpu_memory_fraction: 0.8
```

## 🚀 Deployment Steps

### 1. System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-dev build-essential curl

# Setup user and directories
sudo useradd -r -s /bin/false protein-diffusion
sudo mkdir -p /opt/protein-diffusion /data/protein-diffusion
sudo chown protein-diffusion:protein-diffusion /opt/protein-diffusion /data/protein-diffusion
```

### 2. Application Deployment
```bash
# Copy application files
sudo cp -r . /opt/protein-diffusion/
cd /opt/protein-diffusion

# Install dependencies as application user
sudo -u protein-diffusion python3 -m venv venv
sudo -u protein-diffusion ./venv/bin/pip install -r requirements.txt
```

### 3. Service Configuration
Create `/etc/systemd/system/protein-diffusion.service`:
```ini
[Unit]
Description=Protein Diffusion Design Lab
After=network.target

[Service]
Type=exec
User=protein-diffusion
Group=protein-diffusion
WorkingDirectory=/opt/protein-diffusion
Environment=PATH=/opt/protein-diffusion/venv/bin
EnvironmentFile=/opt/protein-diffusion/config/.env
ExecStart=/opt/protein-diffusion/venv/bin/streamlit run app.py
Restart=always
RestartSec=10

# Security
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/data/protein-diffusion

[Install]
WantedBy=multi-user.target
```

### 4. Start Services
```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable protein-diffusion
sudo systemctl start protein-diffusion

# Check status
sudo systemctl status protein-diffusion
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Install certbot
sudo apt install certbot

# Obtain SSL certificate
sudo certbot certonly --standalone -d your-domain.com

# Configure nginx reverse proxy
sudo cp deployment/nginx/nginx.conf /etc/nginx/sites-available/protein-diffusion
sudo ln -s /etc/nginx/sites-available/protein-diffusion /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Firewall Configuration
```bash
# Configure UFW
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### API Security
```bash
# Generate API keys
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Set up authentication
export PROTEIN_DIFFUSION_API_KEY=your-generated-key
export PROTEIN_DIFFUSION_ENABLE_AUTH=true
```

## 📊 Monitoring Setup

### System Monitoring
```bash
# Install monitoring stack
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Access dashboards
# Grafana: http://your-domain:3000
# Prometheus: http://your-domain:9090
```

### Application Monitoring
```bash
# Enable built-in monitoring
export PROTEIN_DIFFUSION_ENABLE_METRICS=true
export PROTEIN_DIFFUSION_METRICS_PORT=9090

# Start enhanced monitoring
python3 -c "
from src.protein_diffusion.enhanced_monitoring import start_monitoring
start_monitoring()
print('Enhanced monitoring started')
"
```

### Health Checks
```bash
# Configure health check endpoint
curl http://localhost:8501/health

# Set up automated monitoring
crontab -e
# Add: */5 * * * * curl -f http://localhost:8501/health || systemctl restart protein-diffusion
```

## 🔧 Performance Tuning

### CPU Optimization
```bash
# Set CPU governor to performance
sudo cpupower frequency-set -g performance

# Configure process affinity
taskset -cp 0-7 $(pgrep -f streamlit)
```

### Memory Optimization  
```bash
# Configure swap
sudo swapon --show
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### GPU Optimization
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Set GPU persistence mode
sudo nvidia-smi -pm 1

# Configure GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## 📈 Scaling Configuration

### Horizontal Scaling
```bash
# Deploy multiple instances
for i in {1..3}; do
  docker run -d \
    --name protein-diffusion-$i \
    -p $((8500+i)):8501 \
    protein-diffusion:latest
done

# Configure load balancer
sudo cp deployment/nginx/load-balancer.conf /etc/nginx/sites-available/
```

### Auto-Scaling with Kubernetes
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: protein-diffusion-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: protein-diffusion
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🛠️ Maintenance

### Daily Tasks
```bash
#!/bin/bash
# /opt/protein-diffusion/scripts/daily-maintenance.sh

# Check system health
systemctl status protein-diffusion

# Clean up logs older than 7 days
find /var/log/protein-diffusion -name "*.log" -mtime +7 -delete

# Update performance statistics
python3 /opt/protein-diffusion/scripts/update-stats.py

# Backup configuration
cp -r /opt/protein-diffusion/config /backup/$(date +%Y%m%d)/
```

### Weekly Tasks
```bash
#!/bin/bash
# /opt/protein-diffusion/scripts/weekly-maintenance.sh

# Clear cache
python3 -c "
from src.protein_diffusion.performance_optimizer import performance_optimizer
performance_optimizer.cache.cleanup_expired()
"

# Optimize database
# Add database optimization commands if using persistent storage

# Security updates
sudo apt update && sudo apt upgrade -y
```

### Backup Strategy
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup/protein-diffusion/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup application
tar -czf $BACKUP_DIR/app.tar.gz /opt/protein-diffusion

# Backup data
tar -czf $BACKUP_DIR/data.tar.gz /data/protein-diffusion

# Backup configuration  
tar -czf $BACKUP_DIR/config.tar.gz /opt/protein-diffusion/config

# Clean old backups (keep 30 days)
find /backup/protein-diffusion -type d -mtime +30 -exec rm -rf {} \;
```

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
sudo journalctl -u protein-diffusion -f

# Check configuration
python3 -m protein_diffusion.cli validate --input test_sequences.txt

# Reset service
sudo systemctl stop protein-diffusion
sudo systemctl reset-failed protein-diffusion
sudo systemctl start protein-diffusion
```

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux | grep streamlit

# Restart service to clear memory
sudo systemctl restart protein-diffusion

# Reduce batch size
export PROTEIN_DIFFUSION_BATCH_SIZE=16
```

#### GPU Issues
```bash
# Check GPU status
nvidia-smi

# Reset GPU state
sudo nvidia-smi --gpu-reset

# Check CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Performance Issues
```bash
# Run performance benchmark
python3 -m protein_diffusion.cli benchmark --output /tmp/benchmark

# Check system resources
htop
iotop -a

# Profile application
python3 -m cProfile -o profile.stats app.py
```

### Log Analysis
```bash
# View application logs
sudo journalctl -u protein-diffusion --since "1 hour ago"

# Monitor error rates
grep "ERROR" /var/log/protein-diffusion/*.log | wc -l

# Performance metrics
grep "performance" /var/log/protein-diffusion/*.log | tail -20
```

## 📞 Support

### Documentation
- **API Documentation**: `/docs/api/`
- **User Guide**: `/docs/user-guide/`
- **Troubleshooting**: `/docs/troubleshooting/`

### Getting Help
- **Issues**: Create issue on GitHub repository
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Report security issues privately

### Monitoring Alerts
Configure alerts for:
- Service downtime
- High error rates (>5%)
- Memory usage >90%
- GPU utilization >95%
- Response time >5 seconds

## ✅ Production Readiness Checklist

### Pre-Deployment
- [ ] All dependencies installed
- [ ] Configuration files reviewed
- [ ] SSL certificates configured
- [ ] Security settings enabled
- [ ] Monitoring setup complete
- [ ] Backup strategy implemented

### Post-Deployment
- [ ] Health checks passing
- [ ] Performance benchmarks met
- [ ] Security scan completed
- [ ] Load testing completed
- [ ] Monitoring alerts configured
- [ ] Documentation updated

### Ongoing
- [ ] Regular security updates
- [ ] Performance monitoring
- [ ] Backup verification
- [ ] Capacity planning
- [ ] Incident response plan

---

## 🎯 Success Criteria

Your deployment is successful when:

1. **Functionality**: All core features working correctly
2. **Performance**: Response times <2 seconds for typical requests
3. **Reliability**: 99.9% uptime SLA met
4. **Security**: All security scans passing
5. **Scalability**: System handles expected load with room for growth
6. **Monitoring**: Complete visibility into system health and performance

---

**🚀 You're now ready for production deployment!**

*Generated by Autonomous SDLC v4.0 - Terragon Labs*