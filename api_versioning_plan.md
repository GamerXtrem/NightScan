# Plan d'amélioration du versioning des APIs NightScan

## Situation actuelle

- **1 API sur 12** a une gestion de version (`/api/v1/`)
- **7 APIs** devraient être versionnées mais ne le sont pas
- **4 APIs** sont internes/opérationnelles et n'ont pas forcément besoin de versioning

## APIs à migrer vers la v1

### 1. **Authentication API** (`auth/auth_routes.py`)
- Actuel : `/api/auth/*`
- Cible : `/api/v1/auth/*`
- Endpoints : login, register, refresh, logout, verify

### 2. **Analytics API** (`analytics_dashboard.py`)
- Actuel : `/analytics/*`
- Cible : `/api/v1/analytics/*`
- Endpoints : metrics, species, zones, exports

### 3. **Cache Management API** (`cache_monitoring.py`)
- Actuel : `/api/cache/*`
- Cible : `/api/v1/cache/*`
- Endpoints : metrics, health, clear

### 4. **Password Reset API** (`web/password_reset.py`)
- Actuel : `/api/password-reset/*`
- Cible : `/api/v1/password-reset/*`
- Endpoints : request, verify, reset

### 5. **Location API** (`nightscan_pi/Program/location_api.py`)
- Actuel : `/api/location/*`
- Cible : `/api/v1/location/*`
- Endpoints : coordinates, history, status, etc.

### 6. **File Management API** (`nightscan_pi/Program/web_api_extensions.py`)
- Actuel : `/api/filename/*`, `/api/files/*`
- Cible : `/api/v1/filename/*`, `/api/v1/files/*`
- Endpoints : parse, generate, statistics

### 7. **Unified Prediction API** (`unified_prediction_system/unified_prediction_api.py`)
- Actuel : `/predict/*`, `/models/*`
- Cible : `/api/v2/predict/*`, `/api/v2/models/*`
- Note : Pourrait être v2 car c'est une nouvelle API unifiée

## Stratégie de migration

### Phase 1 : Préparation (Sans casser l'existant)
1. Créer un middleware de redirection pour maintenir la compatibilité
2. Ajouter les nouvelles routes versionnées en parallèle des anciennes
3. Marquer les anciennes routes comme dépréciées

### Phase 2 : Migration progressive
1. Mettre à jour les clients (iOS app, web frontend) pour utiliser les nouvelles routes
2. Ajouter des headers de dépréciation sur les anciennes routes
3. Logger l'utilisation des anciennes routes

### Phase 3 : Nettoyage
1. Après période de transition (3-6 mois), supprimer les anciennes routes
2. Maintenir uniquement les routes versionnées

## Implémentation proposée

### 1. Middleware de versioning
```python
# api_versioning_middleware.py
def add_versioning_middleware(app):
    """Add middleware to handle API versioning and redirects."""
    
    # Map old routes to new versioned routes
    ROUTE_MAPPINGS = {
        '/api/auth/': '/api/v1/auth/',
        '/analytics/api/': '/api/v1/analytics/',
        '/api/cache/': '/api/v1/cache/',
        '/api/password-reset/': '/api/v1/password-reset/',
        '/api/location/': '/api/v1/location/',
        '/api/filename/': '/api/v1/filename/',
        '/api/files/': '/api/v1/files/',
    }
    
    @app.before_request
    def version_redirect():
        for old_prefix, new_prefix in ROUTE_MAPPINGS.items():
            if request.path.startswith(old_prefix):
                # Add deprecation header
                @after_this_request
                def add_header(response):
                    response.headers['X-API-Deprecation-Warning'] = (
                        f"This endpoint is deprecated. Use {new_prefix} instead."
                    )
                    return response
                
                # Log deprecated usage
                logger.warning(f"Deprecated API call: {request.path}")
```

### 2. Créer un blueprint centralisé pour toutes les APIs v1
```python
# api_v1_consolidated.py
from flask import Blueprint

# Create main v1 blueprint
api_v1_bp = Blueprint('api_v1', __name__, url_prefix='/api/v1')

# Import and register sub-blueprints
from auth.auth_routes import auth_bp_v1
from analytics_dashboard import analytics_bp_v1
from cache_monitoring import cache_bp_v1

api_v1_bp.register_blueprint(auth_bp_v1, url_prefix='/auth')
api_v1_bp.register_blueprint(analytics_bp_v1, url_prefix='/analytics')
api_v1_bp.register_blueprint(cache_bp_v1, url_prefix='/cache')
```

### 3. Configuration centralisée des versions
```python
# api_config.py
API_VERSIONS = {
    'v1': {
        'status': 'stable',
        'deprecated': False,
        'sunset_date': None,
        'endpoints': [
            'auth', 'analytics', 'cache', 'detections',
            'quota', 'retention', 'location', 'files'
        ]
    },
    'v2': {
        'status': 'beta',
        'deprecated': False,
        'sunset_date': None,
        'endpoints': ['predict', 'models']
    }
}
```

## Bénéfices attendus

1. **Évolution contrôlée** : Possibilité de faire évoluer les APIs sans casser les clients existants
2. **Documentation claire** : Version explicite dans l'URL
3. **Dépréciation progressive** : Transition en douceur vers de nouvelles versions
4. **Compatibilité** : Support de plusieurs versions en parallèle
5. **Standardisation** : Structure cohérente pour toutes les APIs

## Timeline proposée

- **Semaine 1-2** : Implémentation du middleware et des routes versionnées
- **Semaine 3-4** : Migration des clients (iOS app, web)
- **Mois 2-3** : Période de transition avec double support
- **Mois 4** : Suppression des anciennes routes non versionnées

Cette approche garantit une migration en douceur tout en améliorant significativement la gestion des APIs.