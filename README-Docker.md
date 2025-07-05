# NightScan Docker & Kubernetes Deployment

This guide explains how to deploy NightScan using Docker and Kubernetes for production environments.

## Quick Start with Docker Compose

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available
- 20GB free disk space

### Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
```bash
# Database
DB_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password

# Security
SECRET_KEY=your-very-secure-secret-key-here
CSRF_SECRET_KEY=your-csrf-secret-key

# Optional: Monitoring
GRAFANA_PASSWORD=your_grafana_password
```

### Launch the Application

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f web
```

### Access the Application

- **Web Application**: http://localhost:8000
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured
- At least 3 worker nodes with 4GB RAM each
- Storage class for persistent volumes

### Quick Deployment

```bash
# Make the deploy script executable
chmod +x scripts/deploy.sh

# Deploy everything
./scripts/deploy.sh deploy

# Check status
./scripts/deploy.sh status
```

### Manual Deployment Steps

1. **Create namespace and secrets:**
```bash
kubectl apply -f k8s/namespace.yaml
```

2. **Deploy infrastructure:**
```bash
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
```

3. **Deploy application:**
```bash
kubectl apply -f k8s/prediction-api.yaml
kubectl apply -f k8s/web-app.yaml
kubectl apply -f k8s/hpa.yaml
```

4. **Deploy monitoring:**
```bash
kubectl apply -f k8s/monitoring.yaml
```

### Environment Variables

Update the secrets in Kubernetes manifests:

```bash
# Generate base64 encoded values
echo -n "your-secret-key" | base64
echo -n "your-db-password" | base64
```

Then update the Secret manifests in `k8s/` files.

## Production Configuration

### Security Considerations

1. **Use strong secrets**: Generate cryptographically secure keys
2. **Enable TLS**: Configure SSL certificates in ingress
3. **Network policies**: Restrict pod-to-pod communication
4. **RBAC**: Configure proper service account permissions

### Scaling Configuration

The deployment includes Horizontal Pod Autoscalers (HPA):

- **Web App**: 3-10 replicas based on CPU/memory
- **Prediction API**: 2-8 replicas based on load

### Storage Requirements

- **PostgreSQL**: 20GB persistent volume
- **Redis**: 5GB persistent volume  
- **Uploads**: 50GB shared volume
- **Models**: 10GB read-only volume
- **Monitoring**: 30GB total (Prometheus + Grafana)

### Resource Limits

**Web Application per pod:**
- CPU: 500m-1000m
- Memory: 1Gi-2Gi

**Prediction API per pod:**
- CPU: 1000m-2000m
- Memory: 2Gi-4Gi

**Database:**
- CPU: 500m-1000m
- Memory: 1Gi-2Gi

## Monitoring & Observability

### Metrics

The application exposes Prometheus metrics at `/metrics` endpoint:

- Request rates and latencies
- Error rates
- Resource usage
- Custom business metrics

### Alerting Rules

Configured alerts in Prometheus:

- High error rate (>10% for 5 minutes)
- High memory usage (>2GB for 10 minutes)
- Database/API downtime
- Prediction processing delays

### Logs

Structured logging with JSON format:

```bash
# View application logs
kubectl logs -f deployment/web-app -n nightscan

# View prediction API logs
kubectl logs -f deployment/prediction-api -n nightscan
```

## Backup & Disaster Recovery

### Database Backup

```bash
# Create backup
kubectl exec -n nightscan postgres-pod -- pg_dump -U nightscan nightscan > backup.sql

# Restore backup
kubectl exec -i -n nightscan postgres-pod -- psql -U nightscan nightscan < backup.sql
```

### Volume Backup

```bash
# Backup persistent volumes using your cluster's backup solution
# Example with Velero:
velero backup create nightscan-backup --include-namespaces nightscan
```

## Troubleshooting

### Common Issues

1. **Pods stuck in Pending**: Check resource constraints and storage
2. **Database connection errors**: Verify secrets and network connectivity  
3. **Image pull errors**: Check image tags and registry access
4. **Out of memory**: Increase resource limits or optimize model loading

### Debug Commands

```bash
# Check pod status
kubectl get pods -n nightscan -o wide

# Describe problematic pod
kubectl describe pod <pod-name> -n nightscan

# Check logs
kubectl logs <pod-name> -n nightscan --previous

# Execute commands in pod
kubectl exec -it <pod-name> -n nightscan -- /bin/bash

# Port forward for debugging
kubectl port-forward -n nightscan service/web-service 8000:8000
```

### Performance Tuning

1. **Database**: Configure PostgreSQL connection pooling
2. **Caching**: Optimize Redis memory settings
3. **Model Loading**: Use model caching and sharing between pods
4. **Network**: Configure ingress controller for optimal performance

## Development Mode

For development with hot reloading:

```bash
# Use development target
docker-compose -f docker-compose.dev.yml up

# Or build development image
docker build --target development -t nightscan-dev .
```

## Migration from Docker Compose to Kubernetes

1. Export data from Docker Compose volumes
2. Update image tags in Kubernetes manifests
3. Create equivalent secrets and configmaps
4. Deploy to Kubernetes following the steps above
5. Import data to new persistent volumes

## Support

For issues with deployment:

1. Check the application logs
2. Verify resource availability
3. Ensure all secrets are properly configured
4. Consult the monitoring dashboards for system health