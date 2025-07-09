# Système d'Entraînement Edge NightScan

Ce système permet d'entraîner des modèles légers optimisés pour l'inférence mobile, séparés des gros modèles serveur.

## 🎯 **Objectifs**

- **Modèles audio ultra-légers** : < 3MB (spectrogrammes)
- **Modèles photo légers** : < 8MB (MobileNetV3)
- **Modèles ultra-légers** : < 1MB (convolutions séparables)
- **Optimisés pour mobile** : TensorFlow Lite + Core ML ready
- **Interface web dédiée** : Entraînement et monitoring séparés

## 📁 **Structure**

```
Edge_Training_System/
├── models/
│   └── lightweight_models.py          # Architectures légères
├── web_interface/
│   ├── edge_training_app.py           # Flask app dédiée
│   ├── templates/
│   │   └── edge_training.html         # Interface web
│   └── static/                        # Assets statiques
├── scripts/
│   └── convert_to_edge.py             # Conversion des modèles
└── README.md                          # Cette documentation
```

## 🚀 **Modèles Disponibles**

### 1. **LightweightAudioModel**
- **Taille** : ~2.5MB
- **Entrée** : Spectrogrammes 128x128
- **Architecture** : CNN léger avec BatchNorm
- **Optimisé pour** : Reconnaissance audio rapide

### 2. **UltraLightAudioModel**
- **Taille** : ~0.8MB
- **Entrée** : Spectrogrammes 128x128
- **Architecture** : Convolutions séparables
- **Optimisé pour** : Dispositifs très contraints

### 3. **LightweightPhotoModel**
- **Taille** : ~7MB
- **Entrée** : Images 224x224
- **Architecture** : MobileNetV3-Small
- **Optimisé pour** : Images nocturnes

## 🔧 **Utilisation**

### Démarrage de l'Interface

```bash
cd Edge_Training_System/web_interface
python edge_training_app.py
```

L'interface sera disponible sur **http://localhost:5002**

### Entraînement Programmatique

```python
from models.lightweight_models import create_lightweight_model

# Créer un modèle audio léger
model = create_lightweight_model('audio', {
    'num_classes': 6,
    'input_size': (128, 128)
})

# Analyser la complexité
from models.lightweight_models import get_model_complexity
complexity = get_model_complexity(model)
print(f"Taille: {complexity['model_size_mb']:.2f}MB")
```

### Conversion vers Mobile

```python
from model_optimization.quantization_pipeline import ModelQuantizationPipeline

# Convertir les modèles
pipeline = ModelQuantizationPipeline()
results = pipeline.quantize_all_models()

# Résultats: TensorFlow Lite + Core ML
```

## 🎨 **Interface Web**

### Fonctionnalités

- **Sélection interactive** des modèles avec métriques
- **Configuration avancée** pour chaque type de modèle
- **Monitoring temps réel** avec WebSocket
- **Graphiques dynamiques** (précision, perte)
- **Téléchargement automatique** des modèles entraînés

### Métriques Affichées

- Nombre de paramètres
- Taille estimée (MB)
- FLOPs estimés
- Compatibilité mobile
- Progression d'entraînement en temps réel

## 📊 **Différences avec les Gros Modèles**

| Aspect | Gros Modèles | Modèles Edge |
|--------|-------------|--------------|
| **Taille** | 50-200MB | 1-8MB |
| **Précision** | 95-98% | 85-92% |
| **Temps d'inférence** | 500-2000ms | 50-300ms |
| **Utilisation** | Serveur | Mobile |
| **Entraînement** | Interface unifiée | Interface dédiée |
| **Optimisation** | Précision maximale | Efficacité mobile |

## 🛠️ **Configuration d'Entraînement**

### Paramètres Optimisés

```python
# Configuration audio léger
{
    'model_type': 'audio',
    'batch_size': 64,  # Plus élevé car modèle plus petit
    'learning_rate': 0.001,
    'epochs': 30,  # Moins d'époques nécessaires
    'optimizer': 'Adam',  # Converge plus vite
    'scheduler': 'ReduceLROnPlateau'
}

# Configuration photo léger
{
    'model_type': 'photo',
    'batch_size': 32,
    'learning_rate': 0.01,
    'epochs': 50,
    'optimizer': 'SGD',  # Meilleur pour MobileNet
    'scheduler': 'StepLR'
}
```

### Augmentation de Données

Les modèles edge utilisent **moins d'augmentation** pour éviter la surcharge :

```python
# Transformations légères
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),  # Réduit
    transforms.RandomRotation(10),           # Limité
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## 🔄 **Workflow Edge vs Serveur**

### 1. **Entraînement Edge**
```
Interface dédiée → Modèles légers → Entraînement optimisé → 
Conversion mobile → Déploiement edge
```

### 2. **Entraînement Serveur**
```
Interface unifiée → Modèles complexes → Entraînement approfondi → 
Optimisation serveur → Déploiement cloud
```

## 📱 **Intégration Mobile**

Les modèles entraînés sont automatiquement compatibles avec :

- **TensorFlow Lite** (Android/iOS)
- **Core ML** (iOS)
- **ONNX Runtime** (Cross-platform)

### Exemple d'Utilisation Mobile

```javascript
// Dans l'app React Native
import { predictWithEdge } from './services/edgePrediction';

// Prédiction avec modèle edge
const result = await predictWithEdge(audioFile, 'audio/wav');
if (result.confidence > 0.8) {
    // Utiliser le résultat edge
    return result;
} else {
    // Fallback vers le serveur
    return await cloudPrediction(audioFile);
}
```

## 🎯 **Cas d'Usage**

### Modèles Audio Edge
- **Détection rapide** d'événements sonores
- **Pré-filtrage** avant envoi serveur
- **Mode offline** pour fonctionnalités de base

### Modèles Photo Edge
- **Classification instantanée** d'images
- **Détection d'objets** en temps réel
- **Prévisualisation** avant analyse complète

## 🔧 **Développement**

### Ajouter un Nouveau Modèle

1. **Définir l'architecture** dans `models/lightweight_models.py`
2. **Ajouter la configuration** dans `edge_training_app.py`
3. **Tester la conversion** avec `quantization_pipeline.py`
4. **Intégrer dans l'interface** web

### Tests

```bash
# Tester tous les modèles
python models/lightweight_models.py

# Tester l'interface
python web_interface/edge_training_app.py

# Tester la conversion
python ../model_optimization/quantization_pipeline.py
```

## 📈 **Performances Attendues**

### Modèles Audio
- **Précision** : 85-90% (vs 95% gros modèles)
- **Vitesse** : 50-150ms (vs 500ms)
- **Taille** : 1-3MB (vs 50MB)

### Modèles Photo
- **Précision** : 88-92% (vs 96% gros modèles)
- **Vitesse** : 100-300ms (vs 800ms)
- **Taille** : 5-8MB (vs 80MB)

## 🚧 **Limitations**

- **Précision réduite** par rapport aux gros modèles
- **Classes limitées** pour maintenir la taille
- **Augmentation limitée** pour éviter la surcharge
- **Nécessite optimisation** selon le dispositif cible

## 🔮 **Roadmap**

- [ ] Support des modèles quantifiés INT8
- [ ] Optimisation spécifique ARM/x86
- [ ] Pruning automatique des modèles
- [ ] Knowledge distillation depuis gros modèles
- [ ] Benchmarks automatisés sur différents devices
- [ ] Pipeline CI/CD pour les modèles edge

---

**Note** : Ce système complète l'écosystème NightScan en fournissant des modèles optimisés pour l'inférence mobile, permettant une expérience utilisateur fluide avec fallback intelligent vers les gros modèles serveur.