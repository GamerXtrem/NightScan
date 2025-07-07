# 📊 Système de Quotas NightScan - Version Simplifiée

## Vue d'ensemble

Le système de quotas NightScan propose un modèle freemium simple avec trois plans :

- **Plan Gratuit** : 600 identifications par mois
- **Plan Premium** : 3000 identifications par mois (19.90€/mois)
- **Plan Entreprise** : 100000 identifications par mois (99.90€/mois)

**Toutes les autres fonctionnalités restent identiques entre les plans.**

## 🏗️ Architecture

### Plans Disponibles

| Plan | Quota mensuel | Prix | Différences |
|------|---------------|------|-------------|
| **Gratuit** | 600 identifications | Gratuit | - |
| **Premium** | 3000 identifications | 19.90€/mois | 5x plus d'identifications |
| **Entreprise** | 100000 identifications | 99.90€/mois | 33x plus que Premium |

### Base de Données

**Tables principales :**
- `plan_features` : Définition des plans (3 plans)
- `user_plans` : Assignation plan utilisateur
- `quota_usage` : Suivi mensuel des quotas
- `quota_transactions` : Audit trail des consommations

## 🚀 API Endpoints

### Gestion des quotas
```bash
# Status quota utilisateur
GET /api/v1/quota/status

# Plans disponibles (gratuit, premium, entreprise)
GET /api/v1/quota/plans  

# Mise à niveau vers premium ou entreprise
POST /api/v1/quota/upgrade {"plan_type": "premium"}
POST /api/v1/quota/upgrade {"plan_type": "enterprise"}

# Analytics d'utilisation de base
GET /api/v1/quota/analytics

# Vérification avant upload
POST /api/v1/quota/check {"file_size_bytes": 1024}
```

### Prédictions avec quotas
```bash
# Prédiction avec vérification automatique des quotas
POST /api/v1/predict
```

## 🔄 Workflow

1. **Nouvel utilisateur** → Plan gratuit automatique (600/mois)
2. **Prédiction** → Vérification quota → Traitement → Consommation quota
3. **Quota atteint** → Message d'upgrade vers premium
4. **Upgrade premium** → 3000 identifications/mois
5. **Upgrade entreprise** → 100000 identifications/mois

## 💾 Migration

```bash
# Migrer les utilisateurs existants
python scripts/migrate-users-to-quotas.py
```

Cette commande :
- Crée les 3 plans (gratuit/premium/entreprise)
- Assigne tous les utilisateurs existants au plan gratuit
- Initialise les quotas pour le mois courant
- Met à jour les compteurs basés sur les prédictions existantes

## 🎯 Exemple d'utilisation

### Vérification du statut quota
```json
GET /api/v1/quota/status
{
  "user_id": 123,
  "plan_type": "free",
  "current_usage": 245,
  "monthly_quota": 600,
  "remaining": 355,
  "usage_percentage": 40.8,
  "reset_date": "2025-08-01T00:00:00Z",
  "days_until_reset": 15
}
```

### Quota dépassé
```json
POST /api/v1/predict
{
  "error": "Quota mensuel dépassé (600/600)",
  "code": "QUOTA_EXCEEDED",
  "quota_status": {
    "current_usage": 600,
    "monthly_quota": 600,
    "plan_type": "free"
  },
  "upgrade_required": true,
  "recommended_plan": "premium"
}
```

### Mise à niveau vers premium
```json
POST /api/v1/quota/upgrade {"plan_type": "premium"}
{
  "success": true,
  "message": "Mis à niveau vers Plan Premium",
  "new_quota": 3000,
  "old_quota": 600
}

POST /api/v1/quota/upgrade {"plan_type": "enterprise"}
{
  "success": true,
  "message": "Mis à niveau vers Plan Entreprise",
  "new_quota": 100000,
  "old_quota": 3000
}
```

## 🛠️ Fonctionnalités

### ✅ Incluses
- Vérification automatique des quotas avant traitement
- Consommation automatique après prédiction réussie
- Réinitialisation mensuelle des quotas
- API complète de gestion des quotas
- Système d'audit des consommations
- Migration automatique des utilisateurs existants

### ❌ Supprimées (par rapport à la version complexe)
- Queue prioritaire
- Analytics avancées (distribution espèces, patterns temporels)
- Différences de taille de fichiers entre plans
- Support email différencié
- Plans multiples complexes

## 🔒 Sécurité

- Vérifications côté serveur uniquement
- Audit trail complet de toutes les consommations
- Protection contre la manipulation côté client
- Intégration avec le système d'authentification existant

---

**Le système est maintenant simple, efficace et prêt pour la production avec une stratégie de monétisation claire :**
- **5x plus d'identifications** pour 19.90€/mois (Premium)  
- **166x plus d'identifications** pour 99.90€/mois (Entreprise)