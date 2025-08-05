# Deployment Guide

This guide covers deploying the Protein Diffusion Design Lab in production environments.

## Quick Start

### Docker Compose (Recommended)

1. **Copy environment file:**
   ```bash
   cp deployment/.env.prod.example deployment/.env.prod
   # Edit deployment/.env.prod with your configuration
   ```

2. **Deploy:**
   ```bash
   cd deployment
   ./scripts/deploy.sh
   ```

3. **Access services:**
   - API: http://localhost/api/
   - Web Interface: http://localhost:8501/
   - Grafana: http://localhost:3000/
   - Prometheus: http://localhost:9090/

## Architecture Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    NGINX    │    │  Streamlit  │    │     API     │
│ Load Balancer│◄──►│Web Interface│◄──►│   Server    │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Redis     │◄──►│ PostgreSQL  │◄──►│ Monitoring  │
│   Cache     │    │  Database   │    │ (Prometheus)│
└─────────────┘    └─────────────┘    └─────────────┘
```

## Environment Configuration

### Required Environment Variables

```bash
# Database
POSTGRES_PASSWORD=your_secure_password
POSTGRES_USER=protein_user
POSTGRES_DB=protein_db

# Security
JWT_SECRET=your_jwt_secret_key
GRAFANA_PASSWORD=admin_password

# Services
REDIS_URL=redis://redis:6379/0
```

### Optional Environment Variables

```bash
# Email alerts
SMTP_USERNAME=your_smtp_user
SMTP_PASSWORD=your_smtp_pass

# Monitoring
WEBHOOK_URL=https://your-webhook.com/alerts

# Performance
MAX_WORKERS=4
REDIS_MAXMEMORY=2gb
```

## Production Deployment Options

### 1. Docker Compose (Single Node)

**Pros:**
- Simple setup and management
- Good for small to medium deployments
- Built-in monitoring and logging

**Cons:**
- Limited scalability
- Single point of failure

**Setup:**
```bash
# Clone repository
git clone https://github.com/your-org/protein-diffusion-design-lab.git
cd protein-diffusion-design-lab

# Configure environment
cp deployment/.env.prod.example deployment/.env.prod
# Edit deployment/.env.prod

# Deploy
deployment/scripts/deploy.sh
```

### 2. Kubernetes (Multi-Node)

**Pros:**
- Highly scalable
- Built-in load balancing
- Auto-healing and rolling updates
- GPU scheduling

**Cons:**
- More complex setup
- Requires Kubernetes expertise

**Setup:**
```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Deploy components
kubectl apply -f deployment/kubernetes/
```

### 3. Cloud Deployment

#### AWS ECS/Fargate
- Use provided Dockerfiles
- Configure ALB for load balancing
- Use RDS for PostgreSQL
- Use ElastiCache for Redis

#### Google Cloud Run
- Serverless deployment option
- Auto-scaling capabilities
- Managed infrastructure

#### Azure Container Instances
- Quick deployment option
- Pay-per-second billing

## Performance Optimization

### Resource Requirements

| Component | CPU | Memory | Storage | GPU |
|-----------|-----|--------|---------|-----|
| API Server | 2 cores | 8GB | 10GB | Optional |
| Web Interface | 1 core | 2GB | 5GB | No |
| Database | 2 cores | 4GB | 50GB | No |
| Redis | 1 core | 2GB | 5GB | No |

### Scaling Guidelines

**Horizontal Scaling:**
- API servers: Scale based on CPU/memory usage
- Database: Use read replicas for read-heavy workloads
- Cache: Redis Cluster for high availability

**Vertical Scaling:**
- Increase memory for larger models
- Add GPUs for faster inference
- Increase storage for model cache

### Performance Tuning

1. **Model Optimization:**
   ```yaml
   model:
     use_jit: true
     use_half_precision: true
     enable_flash_attention: true
   ```

2. **Caching Strategy:**
   ```yaml
   cache:
     sequence_cache_size: 5000
     structure_cache_size: 1000
     expire_time: 3600
   ```

3. **Database Optimization:**
   ```yaml
   database:
     pool_size: 20
     max_overflow: 30
     pool_timeout: 30
   ```

## Security Considerations

### Network Security
- Use HTTPS in production
- Configure firewall rules
- Enable rate limiting
- Use VPN for admin access

### Application Security
- Regular security updates
- Input validation and sanitization
- Authentication and authorization
- Secure API endpoints

### Data Security
- Encrypt data at rest
- Secure database connections
- Regular backups
- Data retention policies

## Monitoring and Alerting

### Metrics Collection
- Application metrics via Prometheus
- System metrics via Node Exporter
- Custom business metrics

### Key Metrics to Monitor
- Response time and throughput
- Error rates and success rates
- Resource utilization (CPU, Memory, GPU)
- Queue lengths and processing times

### Alerting Rules
```yaml
groups:
- name: protein-diffusion
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: High error rate detected
  
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 10
    for: 5m
    annotations:
      summary: High response time detected
```

## Backup and Recovery

### Automated Backups
```bash
# Database backup (daily)
0 2 * * * docker-compose exec postgres pg_dump -U protein_user protein_db > backup.sql

# Model files backup
0 3 * * * tar -czf models_backup.tar.gz /app/models/
```

### Recovery Procedures
1. **Database Recovery:**
   ```bash
   docker-compose exec postgres psql -U protein_user -d protein_db < backup.sql
   ```

2. **Application Recovery:**
   ```bash
   # Rollback to previous version
   deployment/scripts/deploy.sh rollback
   ```

## Troubleshooting

### Common Issues

1. **High Memory Usage:**
   - Reduce model cache size
   - Enable memory optimization
   - Add more RAM or use swap

2. **Slow Response Times:**
   - Check GPU availability
   - Optimize batch processing
   - Scale horizontally

3. **Database Connection Issues:**
   - Check connection pool settings
   - Verify network connectivity
   - Monitor database performance

### Debugging Tools
```bash
# Check container logs
docker-compose logs -f protein-diffusion-api

# Monitor resource usage
docker stats

# Database queries
docker-compose exec postgres psql -U protein_user -d protein_db

# Redis cache status
docker-compose exec redis redis-cli info
```

### Log Locations
- Application logs: `/app/logs/`
- NGINX logs: `/var/log/nginx/`
- Container logs: `docker-compose logs`

## Maintenance

### Regular Tasks
- [ ] Update dependencies monthly
- [ ] Review and rotate secrets quarterly
- [ ] Monitor disk usage weekly
- [ ] Check backup integrity monthly
- [ ] Review performance metrics weekly

### Update Procedures
1. Test updates in staging environment
2. Create backup before updates
3. Use rolling updates to minimize downtime
4. Monitor for issues after deployment

### Health Checks
```bash
# API health
curl -f http://localhost/api/health

# Web interface health
curl -f http://localhost:8501/healthz

# Database health
docker-compose exec postgres pg_isready

# Redis health
docker-compose exec redis redis-cli ping
```

## Cost Optimization

### Resource Management
- Use spot instances for batch processing
- Schedule non-critical workloads during off-peak hours
- Implement auto-scaling policies

### Storage Optimization
- Use object storage for model files
- Implement data lifecycle policies
- Compress backups and logs

### Monitoring Costs
- Set up billing alerts
- Review resource usage regularly
- Optimize based on usage patterns

## Support and Maintenance

### Getting Help
- Check the troubleshooting section first
- Review application logs
- Create detailed issue reports
- Include system information and configurations

### Professional Support
For enterprise deployments, consider:
- Dedicated support channels
- Custom deployment assistance
- Performance optimization consulting
- Training and onboarding services