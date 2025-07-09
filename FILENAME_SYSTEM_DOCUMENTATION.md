# 📁 Système de Nommage des Fichiers NightScan

## 🎯 **Nouveau Format Unifié Implémenté**

### **Format Structure**
```
{TYPE}_{YYYYMMDD}_{HHMMSS}_{LAT}_{LON}.{EXT}
```

### **Exemples Concrets**
- **Audio** : `AUD_20240109_143045_4695_0745.wav`
- **Image** : `IMG_20240109_143045_4695_0745.jpg`
- **Vidéo** : `VID_20240109_143045_4695_0745.mp4`

### **Composants du Format**

| Composant | Description | Exemple | Format |
|-----------|-------------|---------|---------|
| **TYPE** | Type de fichier | `AUD`, `IMG`, `VID` | 3 caractères |
| **DATE** | Date de création | `20240109` | YYYYMMDD |
| **HEURE** | Heure de création | `143045` | HHMMSS |
| **LAT** | Latitude GPS compacte | `4695` | 4 caractères |
| **LON** | Longitude GPS compacte | `0745` | 4 caractères |
| **EXT** | Extension | `wav`, `jpg`, `mp4` | Variable |

## 🗂️ **Fichiers Créés**

### **1. Module Principal**
#### `filename_utils.py`
- **FilenameGenerator** : Génère des noms avec GPS intégré
- **FilenameParser** : Parse tous les formats (ancien et nouveau)
- **Fonctions utilitaires** : `create_audio_filename()`, `create_image_filename()`

### **2. Intégration Système**
#### `main.py` (modifié)
- Utilise `FilenameGenerator` pour l'audio
- Intègre la localisation GPS automatiquement

#### `camera_trigger.py` (modifié)
- Utilise `FilenameGenerator` pour les images
- Format unifié avec l'audio

### **3. Test et Validation**
#### `test_filename_system.py`
- Tests complets du système
- Validation encodage/décodage GPS
- Vérification nouveau format uniquement

## 🔧 **Fonctionnalités Techniques**

### **Encodage GPS Compact**
```python
# Latitude: 46.9480 -> 4695 (4 caractères)
# Longitude: 7.4474 -> 0745 (4 caractères)
```


### **Support Multi-Format**
```python
FILE_TYPES = {
    'audio': 'AUD',
    'image': 'IMG', 
    'video': 'VID'
}
```

## 📊 **Comparaison des Formats**

### **Nouveau Format**
```
Audio: AUD_20240109_143045_4695_0745.wav
Image: IMG_20240109_143045_4695_0745.jpg
```

### **Avantages du Nouveau Format**
1. **📍 GPS intégré** : Localisation visible dans le nom
2. **🎵 Type identifiable** : Distinction audio/photo immédiate
3. **📅 Lisible** : Date/heure compréhensible
4. **🗂️ Tri automatique** : Ordre chronologique et géographique
5. **🔄 Cohérence** : Format unifié pour tous les types

## 🚀 **Utilisation**

### **Génération Automatique**
```python
from filename_utils import FilenameGenerator
from location_manager import location_manager

# Initialisation
generator = FilenameGenerator(location_manager)

# Génération automatique avec GPS
audio_name = generator.generate_audio_filename()
image_name = generator.generate_image_filename()
```

### **Parsing de Fichiers**
```python
from filename_utils import FilenameParser

parser = FilenameParser()

# Parse le nouveau format
parsed = parser.parse_filename("AUD_20240109_143045_4695_0745.wav")

print(f"Type: {parsed['type']}")           # 'audio'
print(f"GPS: {parsed['latitude']}, {parsed['longitude']}")  # 46.95, 7.45
print(f"Date: {parsed['timestamp']}")      # datetime object
```


## 🧪 **Tests et Validation**

### **Exécution des Tests**
```bash
python test_filename_system.py
```

### **Tests Inclus**
- ✅ Génération de noms avec GPS
- ✅ Parsing du nouveau format
- ✅ Encodage/décodage coordonnées GPS
- ✅ Validation précision géographique
- ✅ Intégration système de localisation
- ✅ Opérations fichiers réels


## 📍 **Intégration GPS**

### **Avec Système de Localisation**
```python
# Utilise la position actuelle du Pi
filename = generator.generate_audio_filename()
# Résultat: AUD_20240109_143045_4695_0745.wav
```

### **Avec Coordonnées Personnalisées**
```python
# Utilise des coordonnées spécifiques
filename = generator.generate_audio_filename(
    latitude=47.3769,
    longitude=8.5417
)
# Résultat: AUD_20240109_143045_4738_0854.wav
```

## 🔧 **Fonctions Utilitaires**

### **Fonctions de Compatibilité**
```python
# Fonctions utilitaires simplifiées
from filename_utils import create_audio_filename, create_image_filename

audio_name = create_audio_filename()  # Utilise position actuelle
image_name = create_image_filename()  # Utilise position actuelle
```

## 📋 **Workflow d'Utilisation**

### **1. Démarrage du Service de Localisation**
```bash
cd NightScanPi/Program
python start_location_service.py start
```

### **2. Exécution du Système Principal**
```bash
python main.py  # Utilise automatiquement le nouveau format
```

### **3. Vérification des Fichiers Générés**
```bash
ls data/audio/    # Voir les fichiers AUD_*.wav
ls data/images/   # Voir les fichiers IMG_*.jpg
```


## 🎯 **Objectifs Atteints**

### ✅ **Informations Intégrées**
- **Date et heure** : Format lisible YYYYMMDD_HHMMSS
- **Point GPS** : Coordonnées compactes intégrées
- **Type de fichier** : AUD/IMG pour distinction immédiate

### ✅ **Fonctionnalités Avancées**
- **Génération automatique** avec GPS actuel
- **Parsing du nouveau format** uniquement
- **Tests complets** et validation

### ✅ **Utilisation Pratique**
- **Tri chronologique** : Ordre automatique par date
- **Tri géographique** : Groupement par GPS
- **Identification rapide** : Type visible immédiatement
- **Traçabilité GPS** : Localisation dans le nom

## 🔧 **Maintenance et Support**

### **Logs et Debugging**
- Logging complet dans tous les modules
- Tests automatisés avec rapports JSON

### **Configuration**
- Formats d'extension configurables
- Intégration location manager flexible

### **Évolution**
- Architecture modulaire extensible
- Support futurs types de fichiers
- Validation robuste des coordonnées

---

## 🎉 **Résultat Final**

Le système de nommage des fichiers NightScan intègre maintenant **automatiquement** :
- ✅ **Date et heure** lisibles
- ✅ **Coordonnées GPS** de la position du Pi
- ✅ **Type de fichier** (audio/image) clairement identifié
- ✅ **Format simplifié** sans code de zone
- ✅ **Nouveau projet** sans migration nécessaire

**Format final** : `AUD_20240109_143045_4695_0745.wav`

Tous les fichiers générés par le Pi incluent désormais ces métadonnées directement dans leur nom, facilitant grandement le tri, l'organisation et l'affichage sur la carte !

---

*Système prêt pour production avec format simplifié et validation complète.*