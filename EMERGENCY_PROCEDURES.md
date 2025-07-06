# 🚨 Procédures d'Urgence NightScan VPS Lite

## 📋 Vue d'ensemble

Ce document décrit les procédures d'urgence pour NightScan en production sur VPS Lite. Utilisez ces procédures uniquement en cas de problème critique affectant la disponibilité du service.

---

## 🚨 Situations d'Urgence

### 1. Service Complètement Indisponible
- Application web inaccessible
- API ne répond pas
- Erreur 500/502/503 persistante

### 2. Performance Critique
- RAM > 95% pendant plus de 5 minutes
- CPU > 95% pendant plus de 5 minutes
- Disque > 95%

### 3. Problème de Sécurité
- Tentatives d'intrusion détectées
- Certificats SSL expirés
- Données sensibles exposées

### 4. Corruption de Données
- Base de données corrompue
- Fichiers de configuration invalides
- Services en boucle de crash

---

## 🔧 Commandes de Diagnostic Rapide

### État Général du Système
```bash
# Statut services Docker
docker-compose -f docker-compose.production.yml ps

# Utilisation ressources
htop
docker stats

# Espace disque
df -h

# Logs récents
docker-compose -f docker-compose.production.yml logs --tail=50
```

### Diagnostics Spécifiques
```bash
# Test connectivité web
curl -I https://VOTRE-DOMAINE.com

# Test API
curl -I https://api.VOTRE-DOMAINE.com/health

# Vérifier certificats SSL
openssl s_client -servername VOTRE-DOMAINE.com -connect VOTRE-DOMAINE.com:443

# État firewall
sudo ufw status

# État fail2ban
sudo fail2ban-client status
```

---

## 🚀 Procédures de Rollback

### Option 1: Rollback Rapide (< 2 minutes)
```bash
# Redémarrage immédiat des services
./scripts/rollback-emergency.sh --quick --force
```

### Option 2: Rollback Version Git (< 5 minutes)
```bash
# Rollback vers version stable précédente
./scripts/rollback-emergency.sh --git main

# Rollback vers tag spécifique
./scripts/rollback-emergency.sh --git v1.2.0
```

### Option 3: Rollback depuis Backup (< 10 minutes)
```bash
# Lister backups disponibles
ls -la ~/backups/

# Rollback depuis backup spécifique
./scripts/rollback-emergency.sh --backup backup_20240101_030000.tar.gz
```

### Option 4: Rollback Configuration Seulement
```bash
# Restaurer uniquement la configuration
./scripts/rollback-emergency.sh --config
```

---

## 🛠️ Réparations d'Urgence

### Problème: Services Docker Crashent

#### Diagnostic
```bash
# Vérifier logs d'erreur
docker-compose -f docker-compose.production.yml logs --tail=100 | grep -i error

# Vérifier resources
docker stats --no-stream
```

#### Solution
```bash
# Redémarrage sélectif
docker-compose -f docker-compose.production.yml restart web
docker-compose -f docker-compose.production.yml restart prediction-api

# Redémarrage complet si nécessaire
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d
```

### Problème: Mémoire Saturée (> 95%)

#### Solution Immédiate
```bash
# Arrêter monitoring temporairement
docker-compose -f docker-compose.monitoring.yml down

# Nettoyer caches
docker system prune -f

# Redémarrer avec limites réduites
docker-compose -f docker-compose.production.yml restart
```

### Problème: Disque Plein (> 95%)

#### Solution Immédiate
```bash
# Nettoyer logs Docker
docker system prune -a -f

# Nettoyer backups anciens
find ~/backups/ -name "*.tar.gz" -mtime +7 -delete

# Nettoyer logs système
sudo journalctl --vacuum-time=1d
```

### Problème: SSL Expiré

#### Solution
```bash
# Forcer renouvellement Let's Encrypt
docker-compose exec letsencrypt sh -c "acme.sh --renew -d VOTRE-DOMAINE.com --force"

# Redémarrer nginx
docker-compose -f docker-compose.production.yml restart nginx-proxy
```

### Problème: Base de Données Corrompue

#### Diagnostic
```bash
# Se connecter à PostgreSQL
docker-compose exec postgres psql -U nightscan -d nightscan

# Vérifier intégrité
\l
\dt
```

#### Solution
```bash
# Restaurer depuis backup le plus récent
./scripts/restore-backup.sh backup_20240101_030000.tar.gz

# Ou restart de la DB
docker-compose -f docker-compose.production.yml restart postgres
```

---

## 📞 Contacts d'Urgence

### Escalade Technique
1. **Niveau 1** : Redémarrage automatique des services
2. **Niveau 2** : Rollback vers version stable
3. **Niveau 3** : Restauration depuis backup
4. **Niveau 4** : Contact équipe de développement

### Informations Clés pour Support
- **VPS** : Infomaniak VPS Lite (4GB/2vCPU/50GB)
- **OS** : Ubuntu Server
- **Stack** : Docker Compose, Nginx, PostgreSQL, Redis
- **Monitoring** : Grafana (https://monitoring.DOMAINE.com)
- **Logs** : `/tmp/nightscan_*.log`

---

## 🔍 Monitoring et Alertes

### Dashboards Grafana Critiques
- **Système** : CPU, RAM, Disque
- **Docker** : Containers status, ressources
- **Application** : Uptime, erreurs, performance
- **Sécurité** : Tentatives connexion, firewall

### Seuils d'Alerte
- **RAM** : > 85% (warning), > 95% (critical)
- **CPU** : > 80% (warning), > 95% (critical)
- **Disque** : > 80% (warning), > 95% (critical)
- **Uptime** : < 99% (warning), < 95% (critical)

### Logs à Surveiller
```bash
# Logs applicatifs
docker-compose -f docker-compose.production.yml logs -f

# Logs système
sudo journalctl -f

# Logs sécurité
sudo tail -f /var/log/fail2ban.log
sudo tail -f /var/log/ufw.log
```

---

## 📋 Checklist Post-Incident

### Immédiatement Après Résolution
- [ ] Vérifier tous services fonctionnels
- [ ] Confirmer accès utilisateurs
- [ ] Vérifier données intactes
- [ ] Documenter actions prises
- [ ] Alerter équipe résolution

### Dans les 24h
- [ ] Analyser logs pour cause racine
- [ ] Vérifier backups post-incident
- [ ] Mettre à jour documentation
- [ ] Planifier correctifs préventifs
- [ ] Réviser procédures si nécessaire

### Dans la semaine
- [ ] Rapport incident détaillé
- [ ] Amélioration monitoring
- [ ] Tests procédures d'urgence
- [ ] Formation équipe sur nouveaux points
- [ ] Optimisations préventives

---

## 🛡️ Prévention

### Monitoring Proactif
- Alertes configurées sur tous seuils critiques
- Surveillance 24/7 des métriques clés
- Tests réguliers des procédures d'urgence
- Backups automatiques et testés

### Maintenance Préventive
- Mises à jour sécurité mensuelles
- Rotation secrets trimestrielle
- Tests de charge semestriels
- Révision configuration annuelle

### Documentation
- Procédures à jour et testées
- Contacts d'urgence vérifiés
- Runbooks détaillés pour chaque scénario
- Formation équipe régulière

---

## 🎯 Objectifs de Récupération

### RTO (Recovery Time Objective)
- **Rollback rapide** : < 2 minutes
- **Rollback version** : < 5 minutes
- **Restauration backup** : < 15 minutes
- **Reconstruction complète** : < 60 minutes

### RPO (Recovery Point Objective)
- **Perte de données max** : < 24 heures
- **Backups automatiques** : Quotidiens
- **Backups testés** : Hebdomadaires
- **Archivage long terme** : Mensuels

---

## 📚 Scripts d'Urgence Disponibles

### Scripts Principaux
- `rollback-emergency.sh` : Rollback complet automatisé
- `test-post-deployment.sh` : Tests validation système
- `backup-production.sh` : Backup manuel immédiat
- `restore-backup.sh` : Restauration depuis backup

### Utilisation Rapide
```bash
# Rollback d'urgence total
./scripts/rollback-emergency.sh --quick --force

# Tests post-réparation
./scripts/test-post-deployment.sh --domain VOTRE-DOMAINE.com

# Backup d'urgence
./scripts/backup-production.sh

# Validation système
python scripts/validate-production.py
```

---

## ⚡ Actions Immédiates par Problème

| Problème | Action Immédiate | Commande |
|----------|------------------|----------|
| Service down | Redémarrage | `./scripts/rollback-emergency.sh --quick` |
| Mémoire pleine | Arrêt monitoring | `docker-compose -f docker-compose.monitoring.yml down` |
| Disque plein | Nettoyage | `docker system prune -a -f` |
| SSL expiré | Renouvellement | `docker-compose restart letsencrypt` |
| DB corrompue | Restauration | `./scripts/restore-backup.sh BACKUP_FILE` |
| Intrusion | Isolation | `sudo ufw deny from IP_ADDRESS` |

---

**🚨 EN CAS D'URGENCE CRITIQUE : Contactez immédiatement l'équipe technique avec les détails du problème et les actions déjà prises.**