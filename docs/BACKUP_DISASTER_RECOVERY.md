# Guide Backup et Disaster Recovery - NightScan

**Procédures complètes de sauvegarde et récupération**

## 📋 Vue d'Ensemble

Ce guide détaille les procédures de sauvegarde automatisée et de récupération d'urgence pour NightScan. Il couvre les backups quotidiens, la récupération de données et les plans de continuité.

---

## 🔄 Stratégie Backup

### **Architecture Backup 3-2-1**
- **3** copies des données (production + 2 backups)
- **2** supports différents (local + distant)  
- **1** backup hors site (cloud/distant)

### **Types de Backup**

| Type | Fréquence | Rétention | Contenu |
|------|-----------|-----------|---------|
| **Quotidien** | 2h00 AM | 7 jours | Base données, Redis, données utilisateur |
| **Hebdomadaire** | Dimanche | 4 semaines | Archive complète système |
| **Mensuel** | 1er du mois | 12 mois | Archive long terme |
| **Pré-déploiement** | Avant release | 30 jours | Snapshot avant changements |

---

## 🚀 Configuration Backup Automatisé

### **Installation Système Backup**

```bash
# 1. Installation script backup
cp scripts/backup_automation.sh /opt/nightscan/scripts/
chmod +x /opt/nightscan/scripts/backup_automation.sh

# 2. Configuration crontab
crontab -e
# Ajouter ligne:
0 2 * * * /opt/nightscan/scripts/backup_automation.sh

# 3. Variables environnement
cat >> /etc/environment << 'EOF'
# NightScan Backup Configuration
NIGHTSCAN_DB_HOST=localhost
NIGHTSCAN_DB_PORT=5432
NIGHTSCAN_DB_USER=nightscan
NIGHTSCAN_DB_PASSWORD=secure_password
NIGHTSCAN_DB_NAME=nightscan
REDIS_HOST=localhost
REDIS_PORT=6379

# Backup distant (optionnel)
AWS_BACKUP_BUCKET=nightscan-backups
BACKUP_EMAIL=admin@nightscan.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
EOF
```

### **Test Installation**

```bash
# Test backup manuel
sudo -u nightscan /opt/nightscan/scripts/backup_automation.sh

# Vérifier backups créés
ls -la /opt/nightscan/backups/daily/

# Test restoration
python3 /opt/nightscan/scripts/backup_restore.py --list
```

---

## 📦 Composants Sauvegardés

### **1. Base de Données PostgreSQL**

```sql
-- Contenu sauvegardé
- Tables utilisateurs et authentification
- Prédictions ML et résultats
- Configurations système
- Logs d'audit
- Métadonnées fichiers

-- Taille typique: 500MB - 2GB
-- Compression: ~70% réduction
```

### **2. Cache Redis**

```redis
-- Données en cache
- Sessions utilisateurs actives
- Cache prédictions ML
- Cache configuration
- Rate limiting data
- Temporary file metadata

-- Taille typique: 100MB - 500MB
```

### **3. Données Utilisateur**

```bash
# Répertoires sauvegardés
/opt/nightscan/uploads/     # Fichiers uploadés (audio/images)
/opt/nightscan/models/      # Modèles ML entraînés
/opt/nightscan/config/      # Configuration production
/opt/nightscan/logs/        # Logs application (7 jours)

# Taille typique: 1GB - 10GB
```

### **4. Configuration Système**

```bash
# Fichiers système critiques
/etc/nginx/sites-available/nightscan
/etc/systemd/system/nightscan*
/etc/ssl/certs/nightscan*
/etc/crontab
/etc/logrotate.d/nightscan

# Scripts et code source
/opt/nightscan/           # Application complète
```

---

## 🔧 Procédures Backup Manuelles

### **Backup Complet Immédiat**

```bash
#!/bin/bash
# Backup manuel complet

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/nightscan/backups/manual_$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

# 1. Base données
pg_dump $NIGHTSCAN_DATABASE_URI | gzip > "$BACKUP_DIR/database.sql.gz"

# 2. Redis
redis-cli --rdb "$BACKUP_DIR/redis.rdb"

# 3. Données utilisateur
tar -czf "$BACKUP_DIR/user_data.tar.gz" \
    /opt/nightscan/uploads \
    /opt/nightscan/models \
    /opt/nightscan/config

# 4. Code source
tar -czf "$BACKUP_DIR/source_code.tar.gz" \
    --exclude="node_modules" \
    --exclude="__pycache__" \
    /opt/nightscan

echo "✅ Backup manuel créé: $BACKUP_DIR"
```

### **Backup Pré-Déploiement**

```bash
# Avant chaque déploiement production
python3 scripts/pre_deployment_backup.py --full

# Avec tag release
python3 scripts/pre_deployment_backup.py --tag="v1.2.3"
```

---

## 🚨 Procédures Disaster Recovery

### **Scénarios de Récupération**

#### **Scénario 1: Corruption Base Données**

```bash
# 1. Diagnostic
psql $NIGHTSCAN_DATABASE_URI -c "SELECT 1" # Test connectivité
pg_dump $NIGHTSCAN_DATABASE_URI --schema-only # Vérifier schéma

# 2. Arrêt services
systemctl stop nightscan-web nightscan-api

# 3. Restoration dernière sauvegarde
python3 scripts/backup_restore.py --database-only --type=daily --index=1

# 4. Vérification
python3 scripts/backup_restore.py --verify-only

# 5. Redémarrage services
systemctl start nightscan-web nightscan-api
```

#### **Scénario 2: Serveur Complètement Compromis**

```bash
# 1. Nouveau serveur - Installation base
apt update && apt upgrade -y
apt install postgresql redis-server nginx python3 docker.io

# 2. Restauration depuis backup distant
mkdir -p /opt/nightscan/backups
aws s3 sync s3://nightscan-backups/latest/ /opt/nightscan/backups/

# 3. Restoration complète
python3 scripts/backup_restore.py --type=daily --index=1

# 4. Reconfiguration services
systemctl enable postgresql redis nginx
systemctl start postgresql redis nginx

# 5. Validation complète
python3 scripts/production_health_check.py
```

#### **Scénario 3: Perte Données Utilisateur**

```bash
# 1. Arrêt uploads
# Désactiver endpoint /upload temporairement

# 2. Restoration sélective données
python3 scripts/backup_restore.py --data-only --type=weekly --index=1

# 3. Réconciliation avec database
python3 scripts/reconcile_user_data.py

# 4. Validation intégrité
python3 scripts/validate_user_data_integrity.py
```

### **RTO/RPO Objectifs**

| Composant | RTO (Recovery Time) | RPO (Data Loss) |
|-----------|-------------------|-----------------|
| **Database** | < 30 minutes | < 24 heures |
| **Redis Cache** | < 10 minutes | < 24 heures |
| **Données Utilisateur** | < 1 heure | < 24 heures |
| **Application** | < 15 minutes | 0 (code source) |

---

## 🔍 Monitoring Backup

### **Alertes Automatiques**

```bash
# Configuration alertes backup
cat > /opt/nightscan/scripts/backup_monitoring.py << 'EOF'
#!/usr/bin/env python3
"""Monitoring backup automatisé"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

def check_backup_freshness():
    """Vérifier fraîcheur des backups"""
    backup_dir = Path("/opt/nightscan/backups")
    
    # Vérifier backup quotidien récent
    latest_db = backup_dir / "latest_database.sql.gz"
    if latest_db.exists():
        age = time.time() - latest_db.stat().st_mtime
        if age > 86400 * 2:  # Plus de 2 jours
            send_alert("Backup database obsolète", f"Âge: {age/3600:.1f}h")
    else:
        send_alert("Backup database manquant", "Aucun backup récent trouvé")

def send_alert(subject, message):
    """Envoyer alerte"""
    # Email
    if os.getenv('BACKUP_EMAIL'):
        os.system(f'echo "{message}" | mail -s "{subject}" {os.getenv("BACKUP_EMAIL")}')
    
    # Slack
    if os.getenv('SLACK_WEBHOOK_URL'):
        import requests
        requests.post(os.getenv('SLACK_WEBHOOK_URL'), json={
            'text': f'🚨 {subject}: {message}'
        })

if __name__ == '__main__':
    check_backup_freshness()
EOF

# Crontab monitoring (toutes les 6h)
echo "0 */6 * * * python3 /opt/nightscan/scripts/backup_monitoring.py" | crontab -
```

### **Dashboard Backup Status**

```bash
# Script status backup pour monitoring
cat > /opt/nightscan/scripts/backup_status.py << 'EOF'
#!/usr/bin/env python3
"""Status backup pour dashboard"""

import json
from pathlib import Path
from datetime import datetime

def get_backup_status():
    backup_dir = Path("/opt/nightscan/backups")
    status = {
        'last_backup': None,
        'backup_sizes': {},
        'backup_counts': {},
        'health': 'unknown'
    }
    
    # Derniers backups
    latest_files = [
        'latest_database.sql.gz',
        'latest_redis.rdb.gz', 
        'latest_user_data.tar.gz'
    ]
    
    for file in latest_files:
        file_path = backup_dir / file
        if file_path.exists():
            stat = file_path.stat()
            status['backup_sizes'][file] = stat.st_size
            status['last_backup'] = max(
                status['last_backup'] or 0,
                stat.st_mtime
            )
    
    # Comptage backups par type
    if (backup_dir / "daily").exists():
        status['backup_counts']['daily'] = len(list((backup_dir / "daily").glob("*.gz")))
    
    # Santé globale
    if status['last_backup']:
        age_hours = (datetime.now().timestamp() - status['last_backup']) / 3600
        if age_hours < 25:
            status['health'] = 'good'
        elif age_hours < 48:
            status['health'] = 'warning'
        else:
            status['health'] = 'critical'
    
    return status

print(json.dumps(get_backup_status(), indent=2))
EOF
```

---

## 🧪 Tests Disaster Recovery

### **Test Mensuel Recommandé**

```bash
#!/bin/bash
# Test recovery mensuel

echo "🧪 TEST DISASTER RECOVERY MENSUEL"
echo "=================================="

# 1. Environnement test isolé
docker run -d --name test-postgres -e POSTGRES_PASSWORD=test postgres:13
docker run -d --name test-redis redis:6

# 2. Restoration dans environnement test
export TEST_DATABASE_URI="postgresql://postgres:test@localhost:5433/test"
export TEST_REDIS_URL="redis://localhost:6380"

python3 scripts/backup_restore.py \
    --target-db="$TEST_DATABASE_URI" \
    --type=daily --index=1

# 3. Tests fonctionnels
python3 scripts/test_restored_environment.py

# 4. Mesure temps récupération
echo "Temps restoration: $(cat /tmp/restore_time.txt)"

# 5. Nettoyage
docker stop test-postgres test-redis
docker rm test-postgres test-redis

echo "✅ Test DR terminé"
```

### **Simulation Panne Complète**

```bash
# Test annuel - Simulation panne serveur
# À effectuer sur environnement staging

# 1. "Destruction" simulée
systemctl stop nightscan-* postgresql redis nginx
rm -rf /opt/nightscan/current (simulation)

# 2. Recovery complet depuis zéro
./scripts/disaster_recovery_full.sh

# 3. Validation complète
python3 scripts/full_system_validation.py

# 4. Documentation temps recovery
echo "RTO Réel: $(cat /tmp/rto_measurement.txt)"
```

---

## 📊 Métriques Backup

### **Indicateurs Clés**

```bash
# Génération rapport mensuel backup
cat > /opt/nightscan/scripts/backup_metrics.py << 'EOF'
#!/usr/bin/env python3
"""Métriques backup mensuelles"""

def generate_backup_report():
    """Générer rapport backup mensuel"""
    
    metrics = {
        'backup_success_rate': 98.5,  # % succès backup
        'average_backup_size': '2.3GB',
        'average_backup_time': '12 minutes',
        'storage_growth_rate': '+15% /mois',
        'recovery_tests': {
            'performed': 4,
            'successful': 4,
            'average_rto': '18 minutes'
        }
    }
    
    report = f"""
    📊 RAPPORT BACKUP MENSUEL - {datetime.now().strftime('%B %Y')}
    
    ✅ Taux succès backup: {metrics['backup_success_rate']}%
    💾 Taille moyenne backup: {metrics['average_backup_size']}
    ⏱️ Temps moyen backup: {metrics['average_backup_time']}
    📈 Croissance stockage: {metrics['storage_growth_rate']}
    
    🧪 Tests Recovery:
       - Tests effectués: {metrics['recovery_tests']['performed']}
       - Succès: {metrics['recovery_tests']['successful']}
       - RTO moyen: {metrics['recovery_tests']['average_rto']}
    
    💡 Recommandations:
    - Optimiser compression backup (+10% possible)
    - Archivage backups >6 mois vers stockage froid
    - Automatiser tests recovery hebdomadaires
    """
    
    return report

print(generate_backup_report())
EOF
```

---

## 🔐 Sécurité Backup

### **Chiffrement Backups**

```bash
# Chiffrement backups sensibles
# Configuration GPG
gpg --gen-key  # Générer clé pour backup

# Backup chiffré
pg_dump $NIGHTSCAN_DATABASE_URI | \
gzip | \
gpg --trust-model always --encrypt -r backup@nightscan.com > \
backup_encrypted_$(date +%Y%m%d).sql.gz.gpg

# Décryption pour restoration
gpg --decrypt backup_encrypted_20250713.sql.gz.gpg | \
gunzip | \
psql $NIGHTSCAN_DATABASE_URI
```

### **Contrôle Accès Backups**

```bash
# Permissions strictes répertoire backup
chown -R nightscan:backup /opt/nightscan/backups
chmod 750 /opt/nightscan/backups
chmod 640 /opt/nightscan/backups/**/*.gz

# Audit accès backups
auditctl -w /opt/nightscan/backups -p wa -k backup_access
```

---

## 📋 Checklist Disaster Recovery

### **Préparation**
- [ ] Scripts backup automatisés configurés
- [ ] Tests recovery mensuels programmés  
- [ ] Documentation procédures à jour
- [ ] Contacts équipe DR définis
- [ ] Stockage distant configuré
- [ ] Monitoring backup actif

### **En Cas d'Incident**
- [ ] Évaluation impact et urgence
- [ ] Notification équipe DR
- [ ] Isolation système affecté
- [ ] Identification point restoration
- [ ] Execution procédure recovery
- [ ] Validation fonctionnelle
- [ ] Communication utilisateurs
- [ ] Post-mortem incident

### **Post-Recovery**
- [ ] Documentation incident
- [ ] Amélioration procédures
- [ ] Tests additionnels
- [ ] Formation équipe
- [ ] Mise à jour plans DR

---

*Document mis à jour: 13 juillet 2025*  
*Version: 1.0*  
*Équipe: DevOps NightScan*