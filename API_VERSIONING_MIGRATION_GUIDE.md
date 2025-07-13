# Guide de migration du versioning des APIs NightScan

## Vue d'ensemble

Ce guide explique comment migrer le système NightScan vers le nouveau système de versioning d'API qui permet une évolution contrôlée et une compatibilité ascendante.

## État actuel vs État cible

### Avant (actuel)
- 1 API sur 12 avec versioning (`/api/v1/`)
- Routes mélangées : `/api/auth/`, `/analytics/`, `/api/cache/`, etc.
- Pas de stratégie de dépréciation
- Difficile de faire évoluer les APIs sans casser les clients

### Après (cible)
- Toutes les APIs avec versioning approprié
- Routes unifiées : `/api/v1/*`, `/api/v2/*`
- Redirections automatiques pour compatibilité
- Headers de dépréciation et migration progressive

## Installation

1. **Vérifier que les nouveaux modules sont présents** :
```bash
ls -la api_versioning/
ls -la api/
```

2. **Installer les dépendances si nécessaire** :
```bash
pip install -r requirements.txt
```

## Intégration dans l'application existante

### Option 1 : Modification minimale de web/app.py

Ajoutez ces lignes dans `web/app.py` après la création de l'app Flask :

```python
# Au début du fichier
from integrate_api_versioning import integrate_versioning_with_app

# Après app = Flask(__name__)
app = Flask(__name__)
app.config.from_object('config')

# Ajouter le versioning AVANT d'enregistrer les blueprints existants
integrate_versioning_with_app(app)

# Continuer avec le reste de la configuration...
```

### Option 2 : Utilisation du décorateur

Modifiez la fonction `create_app` :

```python
from integrate_api_versioning import modify_existing_app_factory

@modify_existing_app_factory
def create_app(config=None):
    app = Flask(__name__)
    # ... configuration existante ...
    return app
```

## Migration des routes

### 1. Routes d'authentification

**Anciennes routes** :
- `/api/auth/login`
- `/api/auth/register`
- `/api/auth/logout`

**Nouvelles routes** :
- `/api/v1/auth/login`
- `/api/v1/auth/register`
- `/api/v1/auth/logout`

Les anciennes routes redirigent automatiquement vers les nouvelles avec un header de dépréciation.

### 2. Routes analytics

**Anciennes routes** :
- `/analytics/dashboard`
- `/analytics/api/metrics`

**Nouvelles routes** :
- `/api/v1/analytics/dashboard`
- `/api/v1/analytics/metrics`

### 3. Nouvelle API v2 (Prédiction unifiée)

**Nouvelles fonctionnalités en v2** :
- `/api/v2/predict/analyze` - Endpoint unifié pour audio et images
- `/api/v2/predict/batch` - Traitement par lots
- `/api/v2/predict/stream` - Streaming temps réel

## Mise à jour des clients

### Application iOS

Dans `ios-app/services/api-gateway.js`, mettez à jour l'URL de base :

```javascript
const config = {
  GATEWAY_URL: process.env.API_GATEWAY_URL || 'http://localhost:8080',
  API_VERSION: 'v1', // Nouvelle config
};

// Mettre à jour les appels API
async login(username, password) {
  const response = await this.makeRequest(`/api/${config.API_VERSION}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Version': config.API_VERSION // Header optionnel
    },
    body: JSON.stringify({ username, password })
  });
}
```

### Frontend Web

Créez un service API centralisé :

```javascript
// api-service.js
class APIService {
  constructor() {
    this.baseURL = '/api/v1';
    this.version = 'v1';
  }
  
  async request(endpoint, options = {}) {
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
        'X-API-Version': this.version
      }
    });
    
    // Vérifier les headers de dépréciation
    if (response.headers.get('X-API-Deprecation-Warning')) {
      console.warn('API Deprecation:', response.headers.get('X-API-Deprecation-Warning'));
    }
    
    return response;
  }
}
```

## Monitoring de la migration

### 1. Vérifier le statut de migration

```bash
curl http://localhost:8000/api/migration-status
```

Réponse :
```json
{
  "total_endpoints": 75,
  "versioned_endpoints": {
    "v1": {"count": 65},
    "v2": {"count": 10}
  },
  "migration_progress": {
    "percentage": 100,
    "status": "complete"
  }
}
```

### 2. Vérifier les routes dépréciées utilisées

```bash
curl http://localhost:8000/api/migration-status | jq .deprecated_endpoint_usage
```

### 3. Logs de dépréciation

Les appels aux anciennes routes sont loggés :
```
WARNING: Deprecated API endpoint used: /api/auth/login -> /api/v1/auth/login (count: 42)
```

## Headers HTTP

### Headers de réponse ajoutés

- `X-API-Version` : Version utilisée pour la requête
- `X-API-Versions-Supported` : Liste des versions supportées
- `X-API-Deprecation-Warning` : Avertissement si route dépréciée
- `X-API-Replacement-Endpoint` : Nouvelle route à utiliser
- `Sunset` : Date de suppression prévue (RFC 7231)

### Exemple de réponse avec dépréciation

```http
HTTP/1.1 200 OK
X-API-Version: v1
X-API-Versions-Supported: v1, v2
X-API-Deprecation-Warning: This endpoint is deprecated. Use /api/v1/auth/login instead.
X-API-Deprecated-Endpoint: /api/auth/login
X-API-Replacement-Endpoint: /api/v1/auth/login
Sunset: Wed, 01 Jul 2024 00:00:00 GMT
```

## Timeline de migration recommandée

### Phase 1 : Déploiement (Semaine 1)
- ✅ Déployer le nouveau système avec redirections
- ✅ Vérifier que toutes les routes fonctionnent
- ✅ Monitorer les erreurs

### Phase 2 : Migration des clients (Semaines 2-4)
- 📱 Mettre à jour l'app iOS
- 💻 Mettre à jour le frontend web
- 📊 Suivre l'utilisation des anciennes routes

### Phase 3 : Communication (Mois 2)
- 📧 Notifier les utilisateurs de l'API
- 📝 Publier la documentation mise à jour
- ⚠️ Activer les warnings dans les logs

### Phase 4 : Dépréciation (Mois 3-6)
- 🚫 Marquer les anciennes routes comme dépréciées
- 📉 Réduire progressivement le support
- 🗑️ Planifier la suppression

## Troubleshooting

### Problème : Routes non trouvées (404)

**Solution** : Vérifiez que le middleware est bien enregistré :
```python
# Dans web/app.py
print(app.before_request_funcs)  # Doit contenir le middleware
```

### Problème : Redirections en boucle

**Solution** : Vérifiez les mappings dans `api_versioning/config.py`

### Problème : Headers manquants

**Solution** : Assurez-vous que le middleware `after_request` est actif

## Support

Pour toute question ou problème :
1. Consultez les logs de l'application
2. Vérifiez `/api/migration-status`
3. Consultez la documentation API : `/api/v1/docs`

## Prochaines étapes

1. ✅ Intégrer le versioning dans l'app
2. ✅ Tester toutes les routes
3. 📱 Migrer les clients
4. 📊 Monitorer l'adoption
5. 🎉 Supprimer les anciennes routes (après 6 mois)