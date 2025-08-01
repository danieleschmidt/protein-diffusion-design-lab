# Deployment Guide

This document outlines deployment strategies for the protein-diffusion-design-lab project.

## 🚀 Deployment Options

### 1. Local Development Deployment

```bash
# Quick local setup
git clone https://github.com/yourusername/protein-diffusion-design-lab.git
cd protein-diffusion-design-lab
conda create -n protein-diffusion python=3.9
conda activate protein-diffusion
pip install -e .[dev]

# Run locally
streamlit run app.py
```

### 2. Docker Deployment

#### Single Container
```bash
# Build and run
docker-compose up protein-diffusion

# Access at http://localhost:8501
```

#### Development with Hot Reload
```bash
# Development mode with volume mounting
docker-compose --profile dev up protein-diffusion-dev
```

#### Production Deployment
```bash
# Production configuration
docker-compose -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose -f docker-compose.prod.yml -f monitoring/docker-compose.monitoring.yml up -d
```

### 3. Cloud Deployment

#### AWS Deployment

**Prerequisites:**
- AWS CLI configured
- Docker installed
- Terraform installed (optional)

**ECS Deployment:**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-west-2.amazonaws.com

docker build -t protein-diffusion .
docker tag protein-diffusion:latest YOUR_ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/protein-diffusion:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/protein-diffusion:latest

# Deploy to ECS (using AWS CLI or Console)
```

**EC2 Deployment:**
```bash
# Launch EC2 instance with Docker
# Instance requirements: GPU-enabled (p3.2xlarge or similar)
# Minimum 16GB RAM, 100GB storage

# SSH to instance and run:
git clone https://github.com/yourusername/protein-diffusion-design-lab.git
cd protein-diffusion-design-lab
docker-compose up -d
```

#### Google Cloud Platform

**Cloud Run Deployment:**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/protein-diffusion
gcloud run deploy --image gcr.io/PROJECT_ID/protein-diffusion --platform managed --memory 16Gi
```

**GKE Deployment:**
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: protein-diffusion
spec:
  replicas: 2
  selector:
    matchLabels:
      app: protein-diffusion
  template:
    metadata:
      labels:
        app: protein-diffusion
    spec:
      containers:
      - name: protein-diffusion
        image: gcr.io/PROJECT_ID/protein-diffusion:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
        ports:
        - containerPort: 8501
```

### 4. High-Availability Production Setup

#### Architecture Overview
```
Internet → Load Balancer → App Instances → Model Storage
                       → Monitoring Stack
                       → Database (Optional)
```

#### Docker Compose Production
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  protein-diffusion:
    build: .
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
    volumes:
      - model_weights:/app/weights:ro
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - protein-diffusion

volumes:
  model_weights:
    driver: local
```

## 🔧 Configuration Management

### Environment Variables

```bash
# Required
CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_CACHE_DIR=/app/weights
TEMP_DIR=/tmp/protein_diffusion

# Optional
LOG_LEVEL=INFO
MAX_BATCH_SIZE=32
ENABLE_MONITORING=true
METRICS_PORT=9090
```

### Model Weights Management

```bash
# Download and cache model weights
python scripts/download_weights.py --model boltz-1b --cache-dir /app/weights

# For production, mount weights as read-only volume
docker run -v /host/weights:/app/weights:ro protein-diffusion
```

## 📊 Monitoring and Observability

### Health Checks

```python
# health_check.py
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': __version__,
        'gpu_available': torch.cuda.is_available(),
        'model_loaded': model is not None
    }
```

### Metrics Collection

```yaml
# Prometheus metrics exposed on :9090/metrics
protein_diffusion_requests_total
protein_diffusion_request_duration_seconds
protein_diffusion_model_inference_time_seconds
protein_diffusion_gpu_memory_usage_bytes
protein_diffusion_active_sessions
```

### Logging Configuration

```yaml
# logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/protein-diffusion.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    level: INFO
    formatter: default
loggers:
  protein_diffusion:
    level: INFO
    handlers: [console, file]
```

## 🔒 Security Considerations

### SSL/TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://protein-diffusion:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Authentication (Optional)

```python
# For production deployments requiring authentication
@app.middleware("http")
async def authenticate(request: Request, call_next):
    if not request.headers.get("Authorization"):
        return Response("Unauthorized", status_code=401)
    return await call_next(request)
```

## 🔄 CI/CD Integration

### Automated Deployment Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        # Deploy to cloud provider
        # Update container images
        # Run health checks
        # Notify team
```

## 🆘 Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Adjust batch size
   export MAX_BATCH_SIZE=16
   ```

2. **Model Loading Failures**
   ```bash
   # Verify model weights
   ls -la /app/weights/
   
   # Re-download if corrupted
   python scripts/download_weights.py --force
   ```

3. **Performance Issues**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Check application logs
   docker logs protein-diffusion
   ```

### Rollback Procedures

```bash
# Quick rollback to previous version
docker tag protein-diffusion:v1.0.0 protein-diffusion:latest
docker-compose up -d

# Or use specific commit
git checkout <previous-commit>
docker-compose build
docker-compose up -d
```

## 📋 Maintenance Checklist

### Weekly
- [ ] Check system resource usage
- [ ] Verify backup integrity
- [ ] Review security logs
- [ ] Update dependencies (automated)

### Monthly
- [ ] Performance optimization review
- [ ] Security audit
- [ ] Disaster recovery test
- [ ] Documentation updates

### Quarterly
- [ ] Infrastructure cost review
- [ ] Architecture review
- [ ] Team access audit
- [ ] Compliance review