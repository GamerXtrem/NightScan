# Guide Procédures Opérationnelles - NightScan

**Guide complet pour équipe DevOps et support production**

## 📋 Vue d'Ensemble

Ce document définit les procédures opérationnelles standardisées pour NightScan en production. Il couvre la surveillance, la maintenance, les déploiements et la gestion des incidents.

---

## 🚀 Procédures Déploiement

### **Déploiement Production Standard**

```bash
# 1. Préparation environnement
export NIGHTSCAN_ENV=production
export NIGHTSCAN_CONFIG_FILE=/opt/nightscan/config/production.json

# 2. Vérification pré-déploiement
python scripts/pre_deployment_check.py
python scripts/validate_env.py

# 3. Sauvegarde avant déploiement
pg_dump $NIGHTSCAN_DATABASE_URI > backup_pre_deploy_$(date +%Y%m%d_%H%M).sql
redis-cli --rdb backup_redis_$(date +%Y%m%d_%H%M).rdb

# 4. Déploiement avec rollback
docker-compose -f docker-compose.production.yml up -d --no-deps web
sleep 30 && python scripts/health_check.py

# 5. Validation post-déploiement
python scripts/post_deployment_validation.py
```

### **Rollback d'Urgence**

```bash
# Rollback automatique en cas d'échec
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d --scale web=2
python scripts/emergency_rollback.py --version=previous
```

---

## 📊 Surveillance et Monitoring

### **Métriques Critiques à Surveiller**

1. **Application Health**
   - Uptime services > 99.9%
   - Response time < 2s (P95)
   - Error rate < 0.1%
   - Memory usage < 80%

2. **Base de Données**
   - Connexions actives < 90% pool
   - Query time < 100ms (P95)
   - Disk usage < 85%
   - Replication lag < 5s

3. **ML Services**
   - Prediction latency < 5s
   - Model accuracy > 95%
   - Queue length < 100
   - GPU memory < 90%

### **Alertes Configurées**

```yaml
# alerts.yml
groups:
  - name: nightscan_critical
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        annotations:
          summary: "Service {{ $labels.instance }} est down"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
        for: 5m
        annotations:
          summary: "Taux d'erreur élevé: {{ $value }}"
          
      - alert: DatabaseConnectionsHigh
        expr: pg_stat_activity_count > 80
        for: 2m
        annotations:
          summary: "Connexions DB élevées: {{ $value }}"
```

### **Commandes Surveillance**

```bash
# Monitoring en temps réel
watch -n 30 'python scripts/system_status.py'

# Logs application
tail -f logs/nightscan.log | grep ERROR

# Métriques base données
psql $NIGHTSCAN_DATABASE_URI -c "SELECT * FROM pg_stat_activity;"

# Status Redis
redis-cli info memory
redis-cli info replication

# Monitoring GPU (si applicable)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

---

## 🔧 Maintenance Régulière

### **Maintenance Quotidienne**

```bash
#!/bin/bash
# daily_maintenance.sh

# 1. Vérification santé système
python scripts/daily_health_check.py

# 2. Nettoyage logs anciens
find logs/ -name "*.log" -mtime +30 -delete

# 3. Optimisation base données
psql $NIGHTSCAN_DATABASE_URI -c "VACUUM ANALYZE;"

# 4. Nettoyage cache Redis
redis-cli eval "for i=1,#KEYS do redis.call('del',KEYS[i]) end" 0 $(redis-cli keys "cache:*:expired")

# 5. Vérification espace disque
df -h | awk '$5 > 85 {print "ALERT: " $1 " is " $5 " full"}'

# 6. Test connectivity ML services
curl -f http://localhost:8002/health || echo "ML service issue"
```

### **Maintenance Hebdomadaire**

```bash
#!/bin/bash
# weekly_maintenance.sh

# 1. Backup complet
pg_dump $NIGHTSCAN_DATABASE_URI | gzip > weekly_backup_$(date +%Y%m%d).sql.gz

# 2. Rotation logs
logrotate /etc/logrotate.d/nightscan

# 3. Mise à jour sécurité OS
apt update && apt list --upgradable

# 4. Analyse performance
python scripts/performance_analysis.py --period=7days

# 5. Vérification certificats SSL
openssl x509 -in /etc/ssl/certs/nightscan.pem -dates -noout
```

### **Maintenance Mensuelle**

```bash
#!/bin/bash
# monthly_maintenance.sh

# 1. Audit sécurité automatisé
python scripts/security_audit.py --full

# 2. Optimisation base données avancée
psql $NIGHTSCAN_DATABASE_URI -c "REINDEX DATABASE nightscan;"

# 3. Analyse logs sécurité
python scripts/security_log_analysis.py --period=30days

# 4. Test disaster recovery
python scripts/test_backup_restore.py --dry-run

# 5. Mise à jour dépendances
pip-audit --requirement requirements.txt
```

---

## 🚨 Gestion Incidents

### **Classification Incidents**

**🔴 P1 - Critique (< 15 minutes)**
- Service complètement indisponible
- Perte de données
- Faille sécurité majeure

**🟡 P2 - Majeur (< 1 heure)**
- Performance dégradée > 50%
- Fonctionnalité principale cassée
- Erreurs utilisateur fréquentes

**🟢 P3 - Mineur (< 4 heures)**
- Fonctionnalité secondaire impactée
- Performance légèrement dégradée
- Problème cosmétique

### **Procédure Réponse Incident**

```bash
# 1. IDENTIFICATION
# - Alertes automatiques
# - Rapports utilisateurs
# - Monitoring proactif

# 2. ÉVALUATION
python scripts/incident_assessment.py --severity=P1

# 3. ESCALATION
# P1: Notification immédiate équipe on-call
# P2: Notification dans 30 minutes
# P3: Ticket normal

# 4. RÉSOLUTION
# Suivre runbooks spécifiques ci-dessous

# 5. POST-MORTEM
python scripts/generate_incident_report.py --incident-id=INC-2025-001
```

---

## 📚 Runbooks Incidents Spécifiques

### **Service Web Indisponible**

```bash
# Diagnostic
curl -I http://localhost:8000/health
docker ps | grep nightscan
docker logs nightscan_web

# Actions correctives
docker restart nightscan_web
docker-compose -f docker-compose.production.yml up -d web

# Si échec
docker-compose -f docker-compose.production.yml down
docker system prune -f
docker-compose -f docker-compose.production.yml up -d
```

### **Base de Données Lente**

```bash
# Diagnostic
psql $NIGHTSCAN_DATABASE_URI -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
psql $NIGHTSCAN_DATABASE_URI -c "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# Actions correctives
psql $NIGHTSCAN_DATABASE_URI -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction' AND state_change < now() - interval '10 minutes';"
psql $NIGHTSCAN_DATABASE_URI -c "VACUUM ANALYZE;"

# Si critique
systemctl restart postgresql
```

### **Espace Disque Saturé**

```bash
# Diagnostic
df -h
du -sh /var/log/* | sort -hr
du -sh /opt/nightscan/logs/* | sort -hr

# Actions correctives
find /var/log -name "*.log" -mtime +7 -delete
find /opt/nightscan/logs -name "*.log" -mtime +3 -delete
docker system prune -f
journalctl --vacuum-time=24h

# Nettoyage cache applicatif
redis-cli flushdb
rm -rf /tmp/nightscan_cache/*
```

### **ML Service Non Réactif**

```bash
# Diagnostic
curl -f http://localhost:8002/health
nvidia-smi
ps aux | grep python | grep prediction

# Actions correctives
pkill -f "prediction"
docker restart nightscan_ml
python unified_prediction_system/unified_prediction_api.py &

# Vérification modèles
python scripts/validate_ml_models.py
```

### **Faille Sécurité Détectée**

```bash
# ACTIONS IMMÉDIATES
# 1. Isoler système affecté
iptables -A INPUT -s [IP_SUSPECTE] -j DROP

# 2. Collecter preuves
python scripts/security_incident_collector.py --incident-id=SEC-2025-001

# 3. Notification équipe sécurité
python scripts/security_alert.py --severity=critical

# 4. Analyse impact
python scripts/security_impact_analysis.py

# 5. Mitigation
# Appliquer patch si disponible
# Désactiver fonctionnalité vulnérable
# Renforcer monitoring
```

---

## 🔄 Backup et Recovery

### **Stratégie Backup**

```bash
# Backup quotidien automatisé
0 2 * * * /opt/nightscan/scripts/daily_backup.sh

# daily_backup.sh
#!/bin/bash
BACKUP_DIR="/opt/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Base données
pg_dump $NIGHTSCAN_DATABASE_URI | gzip > $BACKUP_DIR/database.sql.gz

# Redis
redis-cli --rdb $BACKUP_DIR/redis.rdb

# Fichiers application
tar -czf $BACKUP_DIR/app_files.tar.gz /opt/nightscan/uploads /opt/nightscan/models

# Configuration
cp -r /opt/nightscan/config $BACKUP_DIR/

# Upload vers S3 (si configuré)
aws s3 sync $BACKUP_DIR s3://nightscan-backups/$(date +%Y%m%d)/

# Nettoyage anciens backups
find /opt/backups -mtime +30 -delete
```

### **Procédure Recovery**

```bash
# 1. Arrêt services
docker-compose -f docker-compose.production.yml down

# 2. Restauration base données
gunzip -c backup_20250713/database.sql.gz | psql $NIGHTSCAN_DATABASE_URI

# 3. Restauration Redis
redis-cli --rdb backup_20250713/redis.rdb
systemctl restart redis

# 4. Restauration fichiers
tar -xzf backup_20250713/app_files.tar.gz -C /

# 5. Restauration configuration
cp -r backup_20250713/config/* /opt/nightscan/config/

# 6. Redémarrage services
docker-compose -f docker-compose.production.yml up -d

# 7. Validation
python scripts/post_recovery_validation.py
```

---

## 📞 Contacts et Escalation

### **Équipe On-Call**

```yaml
Primary:
  - DevOps Lead: +33 X XX XX XX XX
  - Email: devops@nightscan.com
  - Slack: @devops-team

Secondary:
  - Security Lead: +33 X XX XX XX XX
  - Email: security@nightscan.com
  - Slack: @security-team

Management:
  - CTO: +33 X XX XX XX XX
  - Email: cto@nightscan.com
```

### **Matrice Escalation**

| Severity | Initial Response | Escalation L1 | Escalation L2 |
|----------|------------------|---------------|---------------|
| P1       | 0-15 min        | 15-30 min     | 30-60 min     |
| P2       | 0-1 hour        | 1-2 hours     | 2-4 hours     |
| P3       | 0-4 hours       | 4-8 hours     | 8-24 hours    |

---

## 📊 Rapports et Métriques

### **Reporting Automatisé**

```bash
# Rapport quotidien
0 8 * * * python scripts/daily_ops_report.py | mail -s "NightScan Daily Report" devops@nightscan.com

# Rapport hebdomadaire
0 8 * * 1 python scripts/weekly_ops_report.py | mail -s "NightScan Weekly Report" management@nightscan.com

# Métriques temps réel
python scripts/realtime_dashboard.py --port=8080
```

### **KPIs Opérationnels**

- **Uptime**: > 99.9%
- **MTTR** (Mean Time To Recovery): < 15 minutes
- **MTBF** (Mean Time Between Failures): > 30 jours
- **Deployment Success Rate**: > 99%
- **Security Incidents**: 0 critiques/mois

---

## 🛡️ Sécurité Opérationnelle

### **Contrôles d'Accès**

```bash
# Accès serveurs production
# - SSH key authentication uniquement
# - VPN obligatoire
# - Session recording
# - Sudo avec justification

# Rotation secrets automatisée
0 3 * * 0 python scripts/rotate_secrets.py

# Audit logs
tail -f /var/log/auth.log | grep sudo
journalctl -f -u nightscan
```

### **Procédures Urgence Sécurité**

```bash
# En cas de compromission
# 1. Isolation immédiate
iptables -P INPUT DROP
iptables -P OUTPUT DROP

# 2. Collecte forensics
python scripts/forensics_collector.py

# 3. Notification authorities si requis
python scripts/incident_notification.py --type=security
```

---

*Document mis à jour: 13 juillet 2025*  
*Version: 1.0*  
*Équipe: DevOps NightScan*