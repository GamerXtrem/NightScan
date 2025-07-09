# Système de Prédiction Unifiée NightScan

Ce système automatise l'aiguillage des fichiers audio et photo vers les modèles de prédiction appropriés, basé sur le format des fichiers NightScan.

## 🚀 Démarrage Rapide

```bash
# Installer les dépendances
pip install torch numpy flask flask-cors pillow librosa

# Démarrer le système
python start_prediction_system.py

# Ou avec des options personnalisées
python start_prediction_system.py --port 8000 --debug
```

## 📁 Structure du Système

```
unified_prediction_system/
├── file_type_detector.py        # Détection automatique du type de fichier
├── model_manager.py             # Gestion unifiée des modèles audio/photo
├── prediction_router.py         # Routeur d'aiguillage automatique
├── unified_prediction_api.py    # API Flask pour les prédictions
├── web_interface.html           # Interface web
├── start_prediction_system.py  # Script de démarrage
└── README.md                   # Cette documentation
```

## 🔄 Fonctionnement

### 1. Détection Automatique des Fichiers

Le système reconnaît automatiquement le format des fichiers NightScan:
- `AUD_YYYYMMDD_HHMMSS_LAT_LON.wav` → Audio brut
- `AUD_YYYYMMDD_HHMMSS_LAT_LON.npy` → Spectrogramme audio
- `IMG_YYYYMMDD_HHMMSS_LAT_LON.jpg` → Image

### 2. Aiguillage Automatique

```
Fichier → Détection Type → Routeur → Modèle Approprié → Prédiction
```

### 3. Modèles Supportés

- **Audio**: EfficientNet entraîné sur spectrogrammes
- **Photo**: EfficientNet, ResNet, ou Vision Transformer

## 🛠️ Utilisation

### API REST

```bash
# Santé du système
curl http://localhost:5000/health

# Prédiction par upload
curl -X POST -F "file=@AUD_20240109_143045_4695_0745.wav" \
     http://localhost:5000/predict/upload

# Prédiction par chemin
curl -X POST -H "Content-Type: application/json" \
     -d '{"file_path": "/path/to/IMG_20240109_143045_4695_0745.jpg"}' \
     http://localhost:5000/predict/file

# Statistiques
curl http://localhost:5000/stats
```

### Utilisation Programmatique

```python
from unified_prediction_system.prediction_router import predict_file

# Prédiction simple
result = predict_file("AUD_20240109_143045_4695_0745.wav")
print(f"Prédiction: {result['predicted_class']}")
print(f"Confiance: {result['confidence']:.2%}")

# Prédiction avec modèle spécifique
result = predict_file("IMG_20240109_143045_4695_0745.jpg", model_id="custom_photo")
```

### Interface Web

1. Lancez le système: `python start_prediction_system.py`
2. Ouvrez l'interface web (s'ouvre automatiquement)
3. Glissez-déposez un fichier ou utilisez le bouton "Choisir un fichier"
4. Visualisez les résultats en temps réel

## 📊 Formats de Sortie

### Résultat de Prédiction

```json
{
  "success": true,
  "model_type": "audio",
  "predicted_class": "bird_song",
  "confidence": 0.87,
  "top_predictions": [
    {"class": "bird_song", "confidence": 0.87},
    {"class": "mammal_call", "confidence": 0.09},
    {"class": "environmental_sound", "confidence": 0.04}
  ],
  "processing_time": 0.234,
  "file_metadata": {
    "date": "20240109",
    "time": "143045",
    "latitude": 46.95,
    "longitude": 7.45
  }
}
```

## ⚙️ Configuration

### Fichier de Configuration (JSON)

```json
{
  "audio_model": {
    "model_path": "Audio_Training_EfficientNet/models/best_model.pth",
    "config": {
      "model_name": "efficientnet-b1",
      "num_classes": 6,
      "pretrained": true,
      "dropout_rate": 0.3
    },
    "class_names": ["bird_song", "mammal_call", "insect_sound", 
                   "amphibian_call", "environmental_sound", "unknown_species"]
  },
  "photo_model": {
    "model_path": "Picture_Training_Enhanced/models/best_model.pth",
    "config": {
      "model_name": "efficientnet-b1",
      "architecture": "efficientnet",
      "num_classes": 8,
      "pretrained": true,
      "dropout_rate": 0.3
    },
    "class_names": ["bat", "owl", "raccoon", "opossum", "deer", "fox", "coyote", "unknown"]
  }
}
```

## 🔧 Dépendances

### Requises
- `torch` - PyTorch pour l'inférence
- `numpy` - Calculs numériques
- `flask` - API web
- `flask-cors` - Support CORS
- `pillow` - Traitement d'images

### Optionnelles
- `librosa` - Conversion WAV → spectrogramme
- `requests` - Tests API

## 🧪 Tests

### Test du Détecteur de Type

```python
from unified_prediction_system.file_type_detector import FileTypeDetector

detector = FileTypeDetector()
file_obj = detector.detect_file_type("AUD_20240109_143045_4695_0745.wav")
print(f"Type: {file_obj.file_type}")
print(f"Valide: {file_obj.is_valid}")
```

### Test du Routeur

```python
from unified_prediction_system.prediction_router import get_prediction_router

router = get_prediction_router()
result = router.predict_file("test_file.jpg")
print(f"Prédiction: {result['predicted_class']}")
```

## 📈 Monitoring

### Statistiques Système

```bash
# Via API
curl http://localhost:5000/stats

# Via interface web
# Section "État du Système"
```

### Logs

- Fichier: `prediction_system.log`
- Niveau: INFO (ou DEBUG avec `--debug`)
- Rotation automatique des logs

## 🛡️ Sécurité

### Limitations

- Taille max fichier: 50MB
- Extensions autorisées: `.wav`, `.npy`, `.jpg`, `.jpeg`
- Validation des formats de fichiers
- Nettoyage automatique des fichiers temporaires

### CORS

- Configuré pour `localhost:3000` et `127.0.0.1:3000`
- Modifiable dans `unified_prediction_api.py`

## 🚨 Dépannage

### Problèmes Courants

1. **Modèles non trouvés**
   ```bash
   # Vérifier les chemins
   python start_prediction_system.py --check-only
   ```

2. **Erreur de dépendances**
   ```bash
   pip install torch numpy flask flask-cors pillow
   ```

3. **Port déjà utilisé**
   ```bash
   python start_prediction_system.py --port 8000
   ```

4. **Conversion audio échoue**
   ```bash
   pip install librosa
   ```

### Logs de Debug

```bash
python start_prediction_system.py --debug
tail -f prediction_system.log
```

## 📋 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Santé du système |
| `/predict/upload` | POST | Upload + prédiction |
| `/predict/file` | POST | Prédiction par chemin |
| `/predict/batch` | POST | Prédictions en lot |
| `/models/status` | GET | Statut des modèles |
| `/models/preload` | POST | Précharger modèles |
| `/stats` | GET | Statistiques |
| `/supported-formats` | GET | Formats supportés |

## 🔮 Fonctionnalités Futures

- [ ] Support de nouveaux formats audio (MP3, FLAC)
- [ ] Prédictions en temps réel via WebSocket
- [ ] Authentification et autorisation
- [ ] Mise en cache des prédictions
- [ ] Export des résultats (CSV, JSON)
- [ ] Métriques avancées (Prometheus)
- [ ] Déploiement Docker
- [ ] Interface de configuration graphique

## 🤝 Contribution

1. Fork le projet
2. Créez une branche feature (`git checkout -b feature/amazing-feature`)
3. Committez vos changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

## 📄 License

Ce projet fait partie de NightScan et suit la même license.