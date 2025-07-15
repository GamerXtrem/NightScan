# 🚀 Plan de Production VPS Lite - NightScan
**Adapté pour Infomaniak VPS Lite avec Docker Compose**

**Statut Global : 4/10 → Objectif : 8.5/10**  
**Configuration minimale VPS : 4GB RAM, 2 vCPU, 50GB SSD**  
**Délai Total Estimé : 3-4 semaines**  

---

## 📊 Architecture VPS Lite vs Cloud

```
┌─────────────────────────────────────────────────────────────┐
│                     VPS LITE INFOMANIAK                      │
├─────────────────────────────────────────────────────────────┤
│  CPU: 2 vCPU  │  RAM: 4GB  │  SSD: 50GB  │  BW: Illimité   │
├─────────────────────────────────────────────────────────────┤
│                    DOCKER COMPOSE STACK                      │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   NGINX      │   WEB APP    │ PREDICTION   │   MONITORING   │
│   (200MB)    │   (1GB)      │   API (2GB)  │   (500MB)      │
├──────────────┼──────────────┼──────────────┼────────────────┤
│  PostgreSQL  │    Redis     │    Loki      │   Prometheus   │
│   (500MB)    │   (200MB)    │   (300MB)    │   (200MB)      │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

---

## 🔄 Adaptations principales K8s → Docker Compose

| Composant K8s | Alternative Docker | Avantage VPS Lite |
|---------------|-------------------|-------------------|
| Deployments | docker-compose services | -90% overhead |
| ConfigMaps | .env files + configs/ | Simple à gérer |
| Secrets | docker secrets / .env.encrypted | Sécurisé |
| Services | docker networks | Natif Docker |
| Ingress | nginx-proxy | Léger |
| PVC | docker volumes | Persistant |
| HPA | docker-compose scale | Manuel mais simple |
| External Secrets | git-crypt / SOPS | Chiffrement local |

---

## 🔴 PHASE 1 - SÉCURITÉ DOCKER (BLOQUANT)
**Délai : 1-2 semaines | Priorité : MAXIMALE**

### 1.1 Migration secrets K8s → Docker
- [ ] **Tâche** : Créer structure sécurisée pour secrets Docker
  ```bash
  # Structure des secrets
  mkdir -p secrets/{production,staging}
  
  # Installer git-crypt pour chiffrement
  brew install git-crypt  # ou apt-get install git-crypt
  
  # Initialiser git-crypt
  git-crypt init
  git-crypt add-gpg-user YOUR_GPG_KEY_ID
  
  # Créer .env.production chiffré
  cat > secrets/production/.env << EOF
  DB_PASSWORD=$(openssl rand -base64 32)
  REDIS_PASSWORD=$(openssl rand -base64 32)
  SECRET_KEY=$(openssl rand -base64 64)
  CSRF_SECRET_KEY=$(openssl rand -base64 32)
  JWT_SECRET=$(openssl rand -base64 32)
  GRAFANA_PASSWORD=$(openssl rand -base64 16)
  EOF
  
  # Ajouter au .gitattributes
  echo "secrets/**/.env filter=git-crypt diff=git-crypt" >> .gitattributes
  ```
- [ ] **Validation** : Secrets chiffrés dans git, déchiffrés localement
- [ ] **Délai** : 2 jours

### 1.2 Docker Compose sécurisé pour production
- [ ] **Tâche** : Créer docker-compose.production.yml
  ```yaml
  version: '3.13'
  
  services:
    nginx:
      image: nginx:alpine
      container_name: nightscan-nginx
      restart: always
      ports:
        - "80:80"
        - "443:443"
      volumes:
        - ./nginx/conf.d:/etc/nginx/conf.d:ro
        - ./nginx/ssl:/etc/nginx/ssl:ro
        - letsencrypt:/etc/letsencrypt
        - upload_data:/var/www/uploads:ro
      networks:
        - nightscan-net
      depends_on:
        - web
      mem_limit: 200m
      cpus: 0.5
      security_opt:
        - no-new-privileges:true
      read_only: true
      tmpfs:
        - /var/cache/nginx
        - /var/run
  
    web:
      image: ghcr.io/gamerxtrem/nightscan/web:${VERSION:-latest}
      container_name: nightscan-web
      restart: always
      env_file:
        - ./secrets/production/.env
      environment:
        - NIGHTSCAN_ENV=production
        - DATABASE_URL=postgresql://nightscan:${DB_PASSWORD}@postgres:5432/nightscan
        - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
        - PREDICT_API_URL=http://prediction-api:8001/api/predict
      volumes:
        - upload_data:/app/uploads
        - logs_data:/app/logs
      networks:
        - nightscan-net
      depends_on:
        postgres:
          condition: service_healthy
        redis:
          condition: service_healthy
      mem_limit: 1g
      cpus: 1.0
      security_opt:
        - no-new-privileges:true
      user: "1000:1000"
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
        interval: 30s
        timeout: 10s
        retries: 3
  
    prediction-api:
      image: ghcr.io/gamerxtrem/nightscan/prediction-api:${VERSION:-latest}
      container_name: nightscan-ml
      restart: always
      environment:
        - TORCH_DEVICE=cpu
        - MODEL_PATH=/app/models/wildlife_model.pth
        - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      env_file:
        - ./secrets/production/.env
      volumes:
        - ./models:/app/models:ro
        - model_cache:/app/.cache
      networks:
        - nightscan-net
      depends_on:
        redis:
          condition: service_healthy
      mem_limit: 2g
      cpus: 1.5
      security_opt:
        - no-new-privileges:true
      user: "1000:1000"
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:8001/api/health"]
        interval: 30s
        timeout: 10s
        retries: 3
  
    postgres:
      image: postgres:13-alpine
      container_name: nightscan-db
      restart: always
      env_file:
        - ./secrets/production/.env
      environment:
        - POSTGRES_DB=nightscan
        - POSTGRES_USER=nightscan
        - POSTGRES_PASSWORD=${DB_PASSWORD}
      volumes:
        - postgres_data:/var/lib/postgresql/data
        - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
      networks:
        - nightscan-net
      mem_limit: 500m
      cpus: 0.5
      security_opt:
        - no-new-privileges:true
      healthcheck:
        test: ["CMD-SHELL", "pg_isready -U nightscan"]
        interval: 10s
        timeout: 5s
        retries: 5
  
    redis:
      image: redis:6-alpine
      container_name: nightscan-cache
      restart: always
      command: >
        redis-server
        --appendonly yes
        --requirepass ${REDIS_PASSWORD}
        --maxmemory 200mb
        --maxmemory-policy allkeys-lru
      volumes:
        - redis_data:/data
      networks:
        - nightscan-net
      mem_limit: 200m
      cpus: 0.3
      security_opt:
        - no-new-privileges:true
      user: "999:999"
      healthcheck:
        test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
        interval: 10s
        timeout: 5s
        retries: 5
  
  volumes:
    postgres_data:
    redis_data:
    upload_data:
    logs_data:
    model_cache:
    letsencrypt:
  
  networks:
    nightscan-net:
      driver: bridge
      ipam:
        config:
          - subnet: 172.20.0.0/16
  ```
- [ ] **Validation** : Docker Compose avec limites ressources et sécurité
- [ ] **Délai** : 3 jours

### 1.3 Hardening Docker et système
- [ ] **Tâche** : Sécuriser l'environnement Docker
  ```bash
  # Docker daemon.json sécurisé
  cat > /etc/docker/daemon.json << EOF
  {
    "icc": false,
    "log-driver": "json-file",
    "log-opts": {
      "max-size": "10m",
      "max-file": "3"
    },
    "userland-proxy": false,
    "no-new-privileges": true,
    "selinux-enabled": true,
    "userns-remap": "default"
  }
  EOF
  
  # Firewall UFW
  ufw default deny incoming
  ufw default allow outgoing
  ufw allow 22/tcp
  ufw allow 80/tcp
  ufw allow 443/tcp
  ufw --force enable
  
  # Fail2ban pour SSH et Nginx
  apt-get install fail2ban
  cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
  ```
- [ ] **Validation** : Score sécurité Docker Bench > 80%
- [ ] **Délai** : 2 jours

### 1.4 Audit et scan de vulnérabilités
- [ ] **Tâche** : Scanner les images Docker
  ```bash
  # Installer Trivy
  wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add -
  echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | tee -a /etc/apt/sources.list.d/trivy.list
  apt-get update && apt-get install trivy
  
  # Scanner toutes les images
  for image in $(docker-compose -f docker-compose.production.yml config | grep 'image:' | awk '{print $2}'); do
    echo "Scanning $image..."
    trivy image --severity HIGH,CRITICAL $image
  done
  
  # Audit Docker
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    docker/docker-bench-security
  ```
- [ ] **Validation** : Zéro vulnérabilité CRITIQUE
- [ ] **Délai** : 1 jour

**🎯 Critères de validation Phase 1 :**
- ✅ Secrets chiffrés avec git-crypt
- ✅ Docker Compose production sécurisé
- ✅ Limites ressources appliquées
- ✅ Scan vulnérabilités passé
- ✅ Firewall et fail2ban actifs

---

## 🟠 PHASE 2 - INFRASTRUCTURE VPS LITE
**Délai : 1 semaine | Priorité : HAUTE**

### 2.1 SSL/TLS avec Let's Encrypt
- [ ] **Tâche** : Configurer nginx-proxy avec auto-SSL
  ```yaml
  # Ajouter au docker-compose.production.yml
  nginx-proxy:
    image: nginxproxy/nginx-proxy:alpine
    container_name: nginx-proxy
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/tmp/docker.sock:ro
      - ./nginx/certs:/etc/nginx/certs
      - ./nginx/vhost.d:/etc/nginx/vhost.d
      - ./nginx/html:/usr/share/nginx/html
      - letsencrypt:/etc/letsencrypt
    networks:
      - nightscan-net
    environment:
      - DEFAULT_HOST=nightscan.yourdomain.com
  
  letsencrypt-companion:
    image: nginxproxy/acme-companion
    container_name: letsencrypt
    restart: always
    volumes_from:
      - nginx-proxy
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - acme:/etc/acme.sh
    environment:
      - DEFAULT_EMAIL=admin@yourdomain.com
  ```
- [ ] **Configuration Nginx optimisée**
  ```nginx
  # nginx/vhost.d/nightscan.yourdomain.com
  client_max_body_size 100M;
  
  # Security headers
  add_header X-Frame-Options "SAMEORIGIN" always;
  add_header X-Content-Type-Options "nosniff" always;
  add_header X-XSS-Protection "1; mode=block" always;
  add_header Referrer-Policy "strict-origin-when-cross-origin" always;
  add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' wss://nightscan.yourdomain.com" always;
  
  # SSL Configuration
  ssl_protocols TLSv1.2 TLSv1.3;
  ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256;
  ssl_prefer_server_ciphers off;
  
  # HSTS
  add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
  ```
- [ ] **Validation** : SSL A+ sur SSLLabs
- [ ] **Délai** : 1 jour

### 2.2 Monitoring léger avec Loki
- [ ] **Tâche** : Remplacer ELK par Loki + Promtail
  ```yaml
  # monitoring-compose.yml
  loki:
    image: grafana/loki:2.9.0
    container_name: loki
    restart: always
    ports:
      - "3100:3100"
    volumes:
      - ./loki/config.yml:/etc/loki/local-config.yaml:ro
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    networks:
      - nightscan-net
    mem_limit: 300m
    cpus: 0.5
  
  promtail:
    image: grafana/promtail:2.9.0
    container_name: promtail
    restart: always
    volumes:
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - ./promtail/config.yml:/etc/promtail/config.yml:ro
    command: -config.file=/etc/promtail/config.yml
    networks:
      - nightscan-net
    mem_limit: 100m
    cpus: 0.2
  
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: always
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=7d'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    networks:
      - nightscan-net
    mem_limit: 200m
    cpus: 0.3
  
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: always
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    ports:
      - "3000:3000"
    networks:
      - nightscan-net
    mem_limit: 200m
    cpus: 0.3
  ```
- [ ] **Configuration Loki légère**
  ```yaml
  # loki/config.yml
  auth_enabled: false
  
  server:
    http_listen_port: 3100
    grpc_listen_port: 9096
  
  common:
    path_prefix: /loki
    storage:
      filesystem:
        chunks_directory: /loki/chunks
        rules_directory: /loki/rules
    replication_factor: 1
    ring:
      kvstore:
        store: inmemory
  
  schema_config:
    configs:
      - from: 2020-10-24
        store: boltdb-shipper
        object_store: filesystem
        schema: v11
        index:
          prefix: index_
          period: 24h
  
  limits_config:
    retention_period: 168h
    enforce_metric_name: false
    reject_old_samples: true
    reject_old_samples_max_age: 168h
    max_entries_limit_per_query: 5000
  ```
- [ ] **Validation** : Logs centralisés, dashboards fonctionnels
- [ ] **Délai** : 2 jours

### 2.3 Backups automatisés
- [ ] **Tâche** : Script de backup pour VPS
  ```bash
  #!/bin/bash
  # backup-nightscan.sh
  
  BACKUP_DIR="/backup/nightscan"
  DATE=$(date +%Y%m%d_%H%M%S)
  RETENTION_DAYS=7
  
  # Créer répertoire backup
  mkdir -p $BACKUP_DIR
  
  # Backup base de données
  docker exec nightscan-db pg_dump -U nightscan nightscan | gzip > $BACKUP_DIR/db_$DATE.sql.gz
  
  # Backup volumes Docker
  docker run --rm -v nightscan_upload_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/uploads_$DATE.tar.gz -C /data .
  docker run --rm -v nightscan_model_cache:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/models_$DATE.tar.gz -C /data .
  
  # Backup configuration
  tar czf $BACKUP_DIR/config_$DATE.tar.gz docker-compose.production.yml nginx/ prometheus/ grafana/ loki/
  
  # Nettoyer vieux backups
  find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete
  
  # Sync vers stockage externe (optionnel)
  # rclone sync $BACKUP_DIR remote:nightscan-backups/
  ```
- [ ] **Crontab**
  ```bash
  # Backup quotidien à 3h du matin
  0 3 * * * /opt/nightscan/scripts/backup-nightscan.sh >> /var/log/nightscan-backup.log 2>&1
  ```
- [ ] **Validation** : Backups testés et restaurables
- [ ] **Délai** : 1 jour

### 2.4 Optimisation performances VPS
- [ ] **Tâche** : Tuning système pour VPS 4GB
  ```bash
  # /etc/sysctl.d/99-nightscan.conf
  # Network optimizations
  net.core.somaxconn = 65535
  net.ipv4.tcp_max_syn_backlog = 8192
  net.ipv4.ip_local_port_range = 1024 65535
  net.ipv4.tcp_tw_reuse = 1
  net.ipv4.tcp_fin_timeout = 15
  
  # Memory optimizations
  vm.swappiness = 10
  vm.dirty_ratio = 15
  vm.dirty_background_ratio = 5
  
  # File system
  fs.file-max = 2097152
  fs.inotify.max_user_watches = 524288
  
  # Apply with: sysctl -p /etc/sysctl.d/99-nightscan.conf
  ```
- [ ] **Docker cleanup automatique**
  ```bash
  # /etc/cron.daily/docker-cleanup
  #!/bin/bash
  docker system prune -af --volumes --filter "until=24h"
  docker volume prune -f
  ```
- [ ] **Validation** : Utilisation RAM < 3.5GB, CPU < 70%
- [ ] **Délai** : 1 jour

**🎯 Critères de validation Phase 2 :**
- ✅ SSL/TLS automatique fonctionnel
- ✅ Monitoring Loki < 500MB RAM total
- ✅ Backups automatiques testés
- ✅ Performances optimisées pour 4GB

---

## 🟡 PHASE 3 - VALIDATION VPS LITE
**Délai : 3-4 jours | Priorité : IMPORTANTE**

### 3.1 Tests de charge adaptés
- [ ] **Tâche** : Tests réalistes pour VPS 2 vCPU
  ```bash
  # test-load-vps.sh
  #!/bin/bash
  
  # Test API prédiction (10 users, 100 requêtes)
  echo "Test 1: API Prédiction"
  docker run --rm -v $(pwd)/tests:/scripts \
    grafana/k6 run -u 10 -i 100 /scripts/prediction-test.js
  
  # Test web app (25 users, 5 min)
  echo "Test 2: Web Application"
  docker run --rm -v $(pwd)/tests:/scripts \
    grafana/k6 run -u 25 -d 5m /scripts/webapp-test.js
  
  # Monitor resources pendant les tests
  docker stats --no-stream > test-resources.log
  ```
- [ ] **Script K6 optimisé**
  ```javascript
  // tests/prediction-test.js
  import http from 'k6/http';
  import { check, sleep } from 'k6';
  
  export let options = {
    thresholds: {
      http_req_duration: ['p(95)<2000'], // 95% < 2s
      http_req_failed: ['rate<0.1'],     // Error rate < 10%
    },
  };
  
  export default function() {
    const file = open('./sample-audio.wav', 'b');
    const response = http.post('https://nightscan.yourdomain.com/api/predict', {
      file: http.file(file, 'audio.wav'),
    });
    
    check(response, {
      'status is 200': (r) => r.status === 200,
      'has predictions': (r) => JSON.parse(r.body).length > 0,
    });
    
    sleep(1);
  }
  ```
- [ ] **Validation** : <2s latence P95, <10% erreurs
- [ ] **Délai** : 1 jour

### 3.2 Disaster Recovery VPS
- [ ] **Tâche** : Test restauration complète
  ```bash
  # restore-vps.sh
  #!/bin/bash
  
  BACKUP_DIR="/backup/nightscan"
  RESTORE_DATE=$1
  
  if [ -z "$RESTORE_DATE" ]; then
    echo "Usage: $0 YYYYMMDD_HHMMSS"
    exit 1
  fi
  
  # Stop all services
  docker-compose -f docker-compose.production.yml down
  
  # Restore database
  gunzip < $BACKUP_DIR/db_$RESTORE_DATE.sql.gz | \
    docker exec -i nightscan-db psql -U nightscan nightscan
  
  # Restore volumes
  docker run --rm -v nightscan_upload_data:/data -v $BACKUP_DIR:/backup \
    alpine sh -c "cd /data && tar xzf /backup/uploads_$RESTORE_DATE.tar.gz"
  
  # Restart services
  docker-compose -f docker-compose.production.yml up -d
  
  # Verify
  docker-compose ps
  curl -f http://localhost/health || exit 1
  ```
- [ ] **Validation** : Restauration < 30 min
- [ ] **Délai** : 1 jour

### 3.3 Monitoring et alertes
- [ ] **Tâche** : Dashboards Grafana optimisés
  ```json
  {
    "dashboard": {
      "title": "NightScan VPS Lite",
      "panels": [
        {
          "title": "Ressources VPS",
          "targets": [
            {
              "expr": "container_memory_usage_bytes{name=~'nightscan.*'}/1024/1024/1024",
              "legendFormat": "{{name}} RAM"
            },
            {
              "expr": "rate(container_cpu_usage_seconds_total{name=~'nightscan.*'}[5m])*100",
              "legendFormat": "{{name}} CPU%"
            }
          ]
        },
        {
          "title": "API Performance", 
          "targets": [
            {
              "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket{job='prediction-api'})",
              "legendFormat": "P95 Latency"
            }
          ]
        }
      ]
    }
  }
  ```
- [ ] **Alertes critiques**
  ```yaml
  # grafana/alerts.yml
  groups:
    - name: vps_alerts
      rules:
        - alert: HighMemoryUsage
          expr: (sum(container_memory_usage_bytes) / 4294967296) > 0.85
          for: 5m
          annotations:
            summary: "Memory usage > 85% on VPS"
        
        - alert: PredictionAPIDown
          expr: up{job="prediction-api"} == 0
          for: 2m
          annotations:
            summary: "Prediction API is down"
  ```
- [ ] **Validation** : Alertes email/SMS fonctionnelles
- [ ] **Délai** : 1 jour

**🎯 Critères de validation Phase 3 :**
- ✅ Tests charge réussis (2s P95)
- ✅ DR testé < 30 min
- ✅ Monitoring complet < 500MB
- ✅ Alertes configurées

---

## 🟢 PHASE 4 - DÉPLOIEMENT VPS PRODUCTION
**Délai : 2-3 jours | Priorité : LIVRAISON**

### 4.1 Préparation VPS production
- [ ] **Tâche** : Setup initial VPS Infomaniak
  ```bash
  # setup-vps.sh
  #!/bin/bash
  
  # Update system
  apt update && apt upgrade -y
  
  # Install Docker
  curl -fsSL https://get.docker.com | sh
  usermod -aG docker $USER
  
  # Install docker-compose
  curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  chmod +x /usr/local/bin/docker-compose
  
  # Install dependencies
  apt install -y git curl wget ufw fail2ban
  
  # Clone repository
  git clone https://github.com/GamerXtrem/NightScan.git /opt/nightscan
  cd /opt/nightscan
  
  # Setup secrets
  git-crypt unlock
  
  # Pull images
  docker-compose -f docker-compose.production.yml pull
  ```
- [ ] **Validation** : VPS prêt, images pulled
- [ ] **Délai** : 0.5 jour

### 4.2 Déploiement Blue-Green simplifié
- [ ] **Tâche** : Script de déploiement sans downtime
  ```bash
  #!/bin/bash
  # deploy-bluegreen.sh
  
  CURRENT_COLOR=$(docker ps --format "table {{.Names}}" | grep -o "blue\|green" | head -1)
  NEW_COLOR=$([ "$CURRENT_COLOR" = "blue" ] && echo "green" || echo "blue")
  
  echo "Current: $CURRENT_COLOR, Deploying to: $NEW_COLOR"
  
  # Deploy new version
  VERSION=$1 docker-compose -p nightscan-$NEW_COLOR -f docker-compose.production.yml up -d
  
  # Wait for health checks
  echo "Waiting for health checks..."
  sleep 30
  
  # Check health
  if curl -f http://localhost:8080/health; then
    echo "New deployment healthy, switching..."
    
    # Update nginx upstream
    sed -i "s/nightscan-$CURRENT_COLOR/nightscan-$NEW_COLOR/g" /etc/nginx/sites-enabled/nightscan
    nginx -s reload
    
    # Stop old deployment after 60s
    sleep 60
    docker-compose -p nightscan-$CURRENT_COLOR down
    
    echo "Deployment complete!"
  else
    echo "Health check failed, keeping current deployment"
    docker-compose -p nightscan-$NEW_COLOR down
    exit 1
  fi
  ```
- [ ] **Validation** : Zero downtime deployment
- [ ] **Délai** : 1 jour

### 4.3 Monitoring post-déploiement
- [ ] **Tâche** : Surveillance intensive 48h
  ```bash
  # monitor-deployment.sh
  #!/bin/bash
  
  # Log toutes les 5 minutes pendant 48h
  for i in {1..576}; do
    echo "=== Check $i - $(date) ===" >> deployment-monitor.log
    
    # Services status
    docker-compose ps >> deployment-monitor.log
    
    # Resources usage
    docker stats --no-stream >> deployment-monitor.log
    
    # API response time
    time curl -s https://nightscan.yourdomain.com/api/health >> deployment-monitor.log
    
    # Disk usage
    df -h >> deployment-monitor.log
    
    echo "" >> deployment-monitor.log
    sleep 300
  done
  ```
- [ ] **Checklist production**
  - [ ] DNS configuré
  - [ ] SSL vérifié (A+ SSLLabs)
  - [ ] Backups testés
  - [ ] Monitoring actif
  - [ ] Alertes configurées
  - [ ] Documentation mise à jour
- [ ] **Validation** : 48h stable, SLA respecté
- [ ] **Délai** : 2 jours

**🎯 Critères de validation Phase 4 :**
- ✅ Déploiement réussi sans downtime
- ✅ 48h monitoring stable
- ✅ Utilisation ressources < 80%
- ✅ Backups automatiques vérifiés

---

## 📊 Comparaison des ressources

| Ressource | K8s Cloud | VPS Lite | Économie |
|-----------|-----------|----------|----------|
| RAM Total | 8-16 GB | 4 GB | -75% |
| vCPU | 4-8 | 2 | -75% |
| Stockage | 100+ GB | 50 GB | -50% |
| Coût/mois | ~$100-200 | ~$20-40 | -80% |
| Complexité | Haute | Moyenne | -60% |
| Maintenance | Automatique | Manuelle | +100% |

---

## 🚨 Points d'attention VPS Lite

1. **Limitations ressources**
   - Pas de scaling automatique
   - Monitoring ressources critique
   - Optimisation code nécessaire

2. **Maintenance manuelle**
   - Updates OS/Docker
   - Rotation logs
   - Backups vérifiés

3. **Pas de haute disponibilité**
   - Single point of failure
   - Backups externes critiques
   - Plan DR testé régulièrement

4. **Sécurité renforcée**
   - Pas de managed services
   - Firewall manuel
   - Updates sécurité manuels

---

## 📈 Métriques de succès

```
Objectif : 4/10 → 8.5/10

Phase 1 (Sécurité) : +2.5 points
Phase 2 (Infra)    : +1.5 points  
Phase 3 (Tests)    : +0.5 points
Phase 4 (Prod)     : 8.5/10 ✓

Performance cibles:
- API latence P95 < 2s
- Uptime > 99.5%
- RAM usage < 3.5GB
- CPU usage < 70%
- Erreur rate < 1%
```

---

## 🛠️ Scripts utiles

```bash
# Status rapide
docker-compose ps && docker stats --no-stream

# Logs temps réel
docker-compose logs -f --tail=100

# Backup manuel
./scripts/backup-nightscan.sh

# Update images
docker-compose pull && ./deploy-bluegreen.sh latest

# Debug ressources
htop
iotop
docker system df
```

---

*Plan VPS Lite créé le $(date) - Optimisé pour Infomaniak VPS 4GB*