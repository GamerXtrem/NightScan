# 📍 Fonctionnalités de Géolocalisation NightScan

## 🎯 Résumé des Fonctionnalités Implémentées

J'ai créé un système complet de gestion de localisation pour NightScan qui permet de **changer facilement la localisation du Pi via l'application mobile** et **récupérer automatiquement la position du téléphone**.

## 🔧 Composants Créés

### 1. **Backend Pi (Côté Serveur)**

#### `location_manager.py`
- **Gestionnaire principal** de localisation avec base de données SQLite
- **Sauvegarde persistante** des coordonnées
- **Validation automatique** des coordonnées GPS
- **Géocodage inverse** pour obtenir les adresses
- **Détection du fuseau horaire** automatique
- **Historique complet** des changements de position

#### `location_api.py`
- **API REST complète** sur le port 5001
- **8 endpoints** pour gérer la localisation
- **Validation des données** et gestion d'erreurs
- **Support CORS** pour l'app mobile

#### `start_location_service.py`
- **Script de démarrage** pour l'API de localisation
- **Monitoring automatique** du processus
- **Gestion des signaux** pour arrêt propre
- **Installation automatique** des dépendances

### 2. **Frontend Mobile (Application iOS/Android)**

#### `LocationSettingsScreen.js`
- **Écran complet** de configuration de localisation
- **Géolocalisation automatique** du téléphone
- **Saisie manuelle** des coordonnées
- **Carte interactive** pour sélection visuelle
- **Historique des positions** avec sources
- **Validation en temps réel** des coordonnées

#### Permissions et Configuration
- **Permissions iOS/Android** ajoutées dans `app.json`
- **Dépendance expo-location** ajoutée dans `package.json`
- **API service** étendu avec 8 fonctions de localisation

## 🚀 Fonctionnalités Disponibles

### ✅ **Géolocalisation du Téléphone**
- **Permission automatique** : Demande d'autorisation GPS
- **Haute précision** : Utilise GPS avec haute précision
- **Validation de précision** : Rejette les positions > 100m d'erreur
- **Mise à jour en un clic** : Transmet la position au Pi automatiquement

### ✅ **Configuration Manuelle**
- **Saisie de coordonnées** : Latitude/longitude décimales
- **Validation en temps réel** : Vérification des limites (-90/90, -180/180)
- **Carte interactive** : Sélection visuelle sur carte avec marqueurs
- **Adresse automatique** : Géocodage inverse pour obtenir l'adresse

### ✅ **Gestion Avancée**
- **Historique complet** : Suivi de tous les changements avec sources
- **Sources multiples** : Téléphone, manuel, GPS, remise à zéro
- **Sauvegarde persistante** : Base de données SQLite sur le Pi
- **Calculs automatiques** : Mise à jour des heures lever/coucher soleil

### ✅ **Interface Utilisateur**
- **Design moderne** : Interface intuitive avec icônes et couleurs
- **Responsive** : Adapté iOS et Android
- **Feedback utilisateur** : Messages de confirmation et erreurs
- **États de chargement** : Indicateurs visuels pendant traitement

## 🛠️ API REST Complète

### Endpoints Disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/location` | Récupère la localisation actuelle |
| `POST` | `/api/location` | Met à jour la localisation |
| `POST` | `/api/location/phone` | Met à jour depuis le téléphone |
| `GET` | `/api/location/history` | Récupère l'historique |
| `GET` | `/api/location/status` | Statut complet de localisation |
| `POST` | `/api/location/validate` | Valide des coordonnées |
| `POST` | `/api/location/reset` | Remet à la valeur par défaut |
| `GET` | `/api/location/export` | Exporte toutes les données |

### Exemple d'Utilisation

```javascript
// Récupérer la position du téléphone
const location = await Location.getCurrentPositionAsync({
  accuracy: Location.Accuracy.High
});

// Envoyer au Pi
const response = await api.post('/location/phone', {
  latitude: location.coords.latitude,
  longitude: location.coords.longitude,
  accuracy: location.coords.accuracy
});
```

## 📱 Utilisation dans l'App

### 1. **Accès à l'écran**
```javascript
// Depuis n'importe quel écran
navigation.navigate('LocationSettings');
```

### 2. **Géolocalisation automatique**
- Bouton "Utiliser la Position du Téléphone"
- Demande automatique de permission
- Validation de la précision GPS
- Confirmation avant mise à jour

### 3. **Configuration manuelle**
- Champs latitude/longitude
- Validation en temps réel
- Bouton "Carte" pour sélection visuelle
- Sauvegarde automatique

### 4. **Historique et gestion**
- Affichage des 5 dernières positions
- Icônes selon la source (téléphone, manuel, etc.)
- Bouton "Remettre à zéro" pour valeur par défaut
- Export des données

## 🔒 Sécurité et Validation

### Validation des Coordonnées
- **Limites géographiques** : -90/90 latitude, -180/180 longitude
- **Types de données** : Vérification float/int uniquement
- **Précision GPS** : Rejet des positions > 100m d'erreur
- **Sanitization** : Protection contre les injections

### Gestion des Erreurs
- **Réseau** : Gestion des timeouts et erreurs de connexion
- **Permissions** : Gestion des refus de géolocalisation
- **Validation** : Messages d'erreur clairs pour l'utilisateur
- **Fallback** : Valeurs par défaut si problème

## 📊 Base de Données

### Tables Créées

#### `pi_location` (Position actuelle)
```sql
CREATE TABLE pi_location (
    id INTEGER PRIMARY KEY,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    address TEXT DEFAULT '',
    zone TEXT DEFAULT '',
    timezone TEXT DEFAULT '',
    source TEXT DEFAULT 'manual',
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1
);
```

#### `location_history` (Historique)
```sql
CREATE TABLE location_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    address TEXT DEFAULT '',
    zone TEXT DEFAULT '',
    source TEXT DEFAULT 'manual',
    changed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## 🚦 Démarrage et Installation

### 1. **Installation des dépendances mobile**
```bash
cd ios-app
npm install
```

### 2. **Démarrage du service Pi**
```bash
cd NightScanPi/Program
python start_location_service.py start
```

### 3. **Vérification du service**
```bash
# Vérifier le statut
python start_location_service.py status

# Tester l'API
curl http://localhost:5001/api/location/test
```

## 📋 Intégration avec NightScan

### Mise à jour automatique des calculs
- **Heures lever/coucher soleil** : Recalculées automatiquement
- **Fuseau horaire** : Détection automatique selon position
- **Configuration système** : Intégration avec `time_config.py`

### Compatibilité existante
- **API existante** : Fonction `get_current_coordinates()` maintenue
- **Base de données** : Utilisation de SQLite comme le système principal
- **Logging** : Intégration avec le système de logs NightScan

## 🎨 Interface Utilisateur

### Écran Principal
- **Position actuelle** : Affichage avec adresse et source
- **Bouton géolocalisation** : "Utiliser la Position du Téléphone"
- **Saisie manuelle** : Champs latitude/longitude
- **Bouton carte** : Sélection visuelle sur map
- **Historique** : 5 dernières positions avec sources

### Permissions
- **iOS** : `NSLocationWhenInUseUsageDescription` configuré
- **Android** : `ACCESS_FINE_LOCATION` et `ACCESS_COARSE_LOCATION`
- **Demande automatique** : Gestion des permissions dans l'app

## ✅ Résultat Final

### **Objectif atteint** : 
✅ **On peut vraiment changer la localisation du Pi via l'app**
✅ **On peut demander la localisation du téléphone et la sauvegarder**
✅ **Système complet avec validation, historique et intégration**

### **Fonctionnalités bonus** :
- 🗺️ **Carte interactive** pour sélection visuelle
- 📊 **Historique complet** avec sources et dates
- 🔄 **Géocodage inverse** pour obtenir les adresses
- ⏰ **Mise à jour automatique** des calculs astronomiques
- 🔒 **Validation robuste** des coordonnées
- 📱 **Interface moderne** et intuitive

Le système est **prêt à l'emploi** et permet une gestion complète de la localisation du Pi directement depuis l'application mobile !

---

*Toutes les fonctionnalités sont **strictement privées** et fonctionnent uniquement sur votre réseau local.*