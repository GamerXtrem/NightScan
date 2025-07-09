# Interface Web Unifiée NightScan

Interface web moderne pour l'entraînement de modèles audio et photo avec monitoring en temps réel.

## 🌟 Fonctionnalités

### Interface Unifiée
- **Dashboard principal** : Sélection audio/photo avec monitoring temps réel
- **Configurations prédéfinies** : Presets optimisés pour chaque modalité
- **Suivi en temps réel** : WebSocket pour monitoring live
- **Comparaison de modèles** : Analyse comparative des performances

### Modalités Supportées
- **Audio** : Entraînement EfficientNet sur spectrogrammes
- **Photo** : Entraînement multi-architectures (EfficientNet, ResNet, ViT)

## 🚀 Démarrage

### Installation
```bash
# Dépendances Python
pip install flask flask-socketio psutil

# Lancer l'interface
cd Picture_Training_Enhanced/web_interface
python training_app.py
```

### Accès
- Interface principale : http://localhost:5000
- Dashboard audio : http://localhost:5000/audio
- Dashboard photo : http://localhost:5000/photo

## 📊 Pages Disponibles

### 1. Dashboard Principal (`/`)
- Sélection de modalité (audio/photo)
- Progression d'entraînement en temps réel
- Graphiques de performance
- Logs d'entraînement

### 2. Configuration (`/config/<modality>`)
- Presets optimisés par modalité
- Paramètres personnalisables
- Validation des entrées
- Prévisualisation des configurations

### 3. Comparaison (`/comparison`)
- Analyse comparative des modèles
- Métriques de performance
- Recommandations par cas d'usage
- Graphiques d'efficacité

### 4. Pages Spécialisées
- `/audio` : Détails entraînement audio
- `/photo` : Détails entraînement photo

## 🔧 Configuration

### Presets Audio
- `efficientnet_b0_fast` : Développement rapide
- `efficientnet_b1_balanced` : Équilibre performance/temps
- `efficientnet_b2_quality` : Haute qualité

### Presets Photo
- `efficientnet_b0_fast` : Développement rapide
- `efficientnet_b1_balanced` : Équilibre optimal
- `resnet50_quality` : Performance robuste
- `vit_b_16_quality` : Qualité maximale

## 🎯 API Endpoints

### Entraînement
- `POST /api/start_training` : Démarrer l'entraînement
- `POST /api/stop_training` : Arrêter l'entraînement
- `GET /api/training_status` : Statut actuel

### Configuration
- `GET /api/available_configs/<modality>` : Configurations disponibles
- `GET /api/config_details/<modality>/<config>` : Détails configuration

### Système
- `GET /api/system_info` : Informations système
- `GET /api/training_history` : Historique d'entraînement

## 🔄 WebSocket Events

### Événements Client → Serveur
- `join_training` : Rejoindre la room d'entraînement
- `leave_training` : Quitter la room

### Événements Serveur → Client
- `training_progress` : Progression d'entraînement
- `training_log` : Logs d'entraînement
- `training_complete` : Fin d'entraînement
- `training_status` : Statut d'entraînement

## 📱 Interface Responsive

L'interface est optimisée pour :
- **Desktop** : Interface complète avec graphiques
- **Tablet** : Interface adaptée avec navigation simplifiée
- **Mobile** : Interface compacte avec fonctionnalités essentielles

## 🎨 Thèmes Visuels

### Audio (Violet)
- Couleur principale : `#667eea`
- Gradient : `#667eea` → `#764ba2`
- Icône : `fas fa-volume-up`

### Photo (Rose)
- Couleur principale : `#f5576c`
- Gradient : `#f093fb` → `#f5576c`
- Icône : `fas fa-camera`

## 📈 Monitoring

### Métriques Suivies
- **Progression** : Époque actuelle, pourcentage, ETA
- **Performance** : Loss, accuracy train/validation
- **Système** : Utilisation GPU, mémoire, CPU
- **Logs** : Messages temps réel avec niveaux

### Visualisations
- **Graphiques de progression** : Loss et accuracy
- **Anneaux de progression** : Pourcentage d'avancement
- **Tableaux de comparaison** : Métriques par modèle

## 🛡️ Sécurité

- Validation des entrées utilisateur
- Sanitization des paramètres
- Gestion des erreurs robuste
- Timeouts pour éviter les blocages

## 🔧 Personnalisation

### Ajouter une Nouvelle Modalité
1. Créer les presets de configuration
2. Ajouter les endpoints API
3. Créer les templates HTML
4. Mettre à jour le JavaScript

### Modifier les Thèmes
1. Éditer les CSS dans les templates
2. Mettre à jour les classes de couleur
3. Ajuster les gradients et icônes

## 📚 Structure des Fichiers

```
web_interface/
├── training_app.py              # Application Flask principale
├── templates/
│   ├── unified_dashboard.html   # Dashboard principal
│   ├── audio_training.html      # Page audio
│   ├── photo_training.html      # Page photo
│   ├── training_config.html     # Configuration
│   └── model_comparison.html    # Comparaison
├── static/                      # Fichiers statiques
└── README.md                    # Documentation
```

## 🤝 Intégration

### Avec Audio_Training_EfficientNet
- Import automatique des configurations
- Réutilisation des utilitaires existants
- Compatibilité avec les modèles existants

### Avec Picture_Training_Enhanced
- Support complet des nouvelles architectures
- Utilisation des presets avancés
- Intégration des métriques complètes

## 🐛 Dépannage

### Problèmes Courants
1. **Port occupé** : Changer le port dans `training_app.py`
2. **Imports manquants** : Vérifier les chemins Python
3. **WebSocket déconnecté** : Recharger la page

### Logs
- Logs Flask : Console d'exécution
- Logs d'entraînement : Interface web
- Logs système : Fichiers de log

Cette interface web unifiée offre une expérience utilisateur moderne et intuitive pour l'entraînement de modèles audio et photo dans l'écosystème NightScan.