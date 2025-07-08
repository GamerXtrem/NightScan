# 📱 Guide d'intégration WiFi pour l'app iOS NightScan

## Vue d'ensemble

Le système WiFi du Raspberry Pi est entièrement contrôlable depuis l'app iOS. Pas besoin d'interface web séparée - tout passe par l'API REST.

## 🚀 Flux de première installation

### 1. Détection du mode hotspot

Quand l'utilisateur fait un son pour réveiller le Pi :

```swift
// L'app détecte automatiquement si le Pi est en mode hotspot
func checkPiStatus() {
    // Si connecté au WiFi "NightScan-Setup"
    if currentSSID == "NightScan-Setup" {
        // Afficher l'écran de configuration initiale
        showInitialSetupScreen()
    }
}
```

### 2. Configuration initiale depuis l'app

```swift
// 1. Scanner les réseaux disponibles
let networks = await api.get("/wifi/scan")

// 2. L'utilisateur sélectionne un réseau et entre le mot de passe
let networkConfig = WiFiNetwork(
    ssid: selectedNetwork,
    password: userPassword,
    priority: 100,  // Haute priorité pour le WiFi principal
    autoConnect: true
)

// 3. Ajouter le réseau
await api.post("/wifi/networks", body: networkConfig)

// 4. Se connecter au réseau
await api.post("/wifi/connect/\(selectedNetwork)")

// 5. Le Pi quitte le mode hotspot et se connecte au réseau
```

## 📡 Endpoints API disponibles

### Gestion des réseaux

```swift
// Obtenir tous les réseaux configurés
GET /wifi/networks
Response: {
    "networks": [
        {
            "ssid": "WiFi_Maison",
            "priority": 100,
            "auto_connect": true,
            "last_connected": 1704067200.0,
            "notes": "WiFi principal"
        }
    ]
}

// Ajouter un nouveau réseau
POST /wifi/networks
Body: {
    "ssid": "NouveauWiFi",
    "password": "motdepasse",
    "priority": 50,
    "auto_connect": true,
    "hidden": false,
    "notes": "WiFi du bureau"
}

// Supprimer un réseau
DELETE /wifi/networks/{ssid}

// Modifier un réseau
PUT /wifi/networks/{ssid}
Body: {
    "priority": 75,
    "notes": "Priorité augmentée"
}
```

### Connexion et statut

```swift
// Scanner les réseaux disponibles
GET /wifi/scan
Response: {
    "networks": [
        {
            "ssid": "WiFi_1",
            "signal_strength": -45,
            "encrypted": true
        }
    ]
}

// Se connecter à un réseau spécifique
POST /wifi/connect/{ssid}
Body: {
    "force": true  // Forcer même si déjà connecté
}

// Se connecter au meilleur réseau disponible
POST /wifi/connect/best

// Se déconnecter
POST /wifi/disconnect

// Obtenir le statut actuel
GET /wifi/status
Response: {
    "mode": "client",  // ou "hotspot" ou "disabled"
    "connected_network": "WiFi_Maison",
    "ip_address": "192.168.1.42",
    "signal_strength": -52,
    "connection_time": 1704067200.0
}
```

### Mode Hotspot

```swift
// Démarrer le hotspot
POST /wifi/hotspot/start

// Arrêter le hotspot
POST /wifi/hotspot/stop

// Obtenir la configuration du hotspot
GET /wifi/hotspot/config

// Modifier la configuration du hotspot
PUT /wifi/hotspot/config
Body: {
    "ssid": "NightScan-Custom",
    "password": "nouveaumotdepasse",
    "channel": 6
}
```

## 🎨 Écrans suggérés pour l'app

### 1. Écran de configuration initiale

```
┌─────────────────────────┐
│    Configuration WiFi    │
│                         │
│ 🆕 Première utilisation │
│                         │
│ Réseaux disponibles:    │
│ ┌─────────────────────┐ │
│ │ WiFi_Maison    -45  │ │
│ │ Livebox_2534   -62  │ │
│ │ iPhone_John    -71  │ │
│ └─────────────────────┘ │
│                         │
│ [+ Configuration manuelle]│
└─────────────────────────┘
```

### 2. Écran de gestion WiFi (dans les paramètres)

```
┌─────────────────────────┐
│   Paramètres > WiFi     │
│                         │
│ État: Connecté ✅       │
│ Réseau: WiFi_Maison     │
│ IP: 192.168.1.42        │
│                         │
│ Réseaux configurés:     │
│ ┌─────────────────────┐ │
│ │ WiFi_Maison    100  │ │
│ │ iPhone_John     70  │ │
│ │ WiFi_Bureau     50  │ │
│ └─────────────────────┘ │
│                         │
│ [+ Ajouter un réseau]   │
│ [🔍 Scanner]            │
└─────────────────────────┘
```

### 3. Modal d'ajout de réseau

```
┌─────────────────────────┐
│    Ajouter un réseau    │
│                         │
│ SSID: [_______________] │
│                         │
│ Mot de passe:           │
│ [____________________]  │
│                         │
│ Priorité: [50] (0-100)  │
│                         │
│ ☑️ Connexion auto       │
│ ☐ Réseau caché         │
│                         │
│ Notes:                  │
│ [____________________]  │
│                         │
│ [Annuler]  [Ajouter ✓]  │
└─────────────────────────┘
```

## 💡 Logique côté app

### Détection automatique du mode

```swift
class WiFiManager {
    func detectPiMode() async -> PiWiFiMode {
        // 1. Vérifier si on est sur le hotspot NightScan
        if WiFi.currentSSID == "NightScan-Setup" {
            return .hotspot
        }
        
        // 2. Essayer de contacter le Pi sur le réseau actuel
        if let status = try? await api.get("/wifi/status") {
            return status.mode == "hotspot" ? .hotspot : .client
        }
        
        // 3. Scanner les réseaux pour trouver NightScan-Setup
        if WiFi.availableNetworks.contains("NightScan-Setup") {
            return .hotspotAvailable
        }
        
        return .unknown
    }
}
```

### Gestion des erreurs réseau

```swift
func connectToNetwork(ssid: String) async {
    do {
        // 1. Ajouter le réseau si pas déjà fait
        try await api.post("/wifi/networks", body: networkConfig)
        
        // 2. Se connecter
        try await api.post("/wifi/connect/\(ssid)")
        
        // 3. Attendre que le Pi se reconnecte
        showAlert("Connexion en cours...")
        
        // 4. Vérifier le nouveau statut après 5 secondes
        try await Task.sleep(seconds: 5)
        let newStatus = try await api.get("/wifi/status")
        
        if newStatus.connected_network == ssid {
            showSuccess("Connecté à \(ssid)")
        } else {
            showError("Connexion échouée")
        }
        
    } catch {
        // Si on perd la connexion, proposer de se reconnecter au hotspot
        if WiFi.availableNetworks.contains("NightScan-Setup") {
            showAlert("Reconnecter au hotspot NightScan?", 
                     action: { WiFi.connect(to: "NightScan-Setup") })
        }
    }
}
```

## 🔒 Sécurité

### Authentification API

Si vous avez un système d'authentification :

```swift
// Ajouter les headers d'auth à toutes les requêtes WiFi
api.setAuthToken(userToken)

// Ou utiliser l'auth basique si configurée
api.setBasicAuth(username: "admin", password: piPassword)
```

### Validation côté app

```swift
// Valider les entrées avant envoi
func validateNetworkConfig(_ config: WiFiNetwork) -> Bool {
    // SSID non vide
    guard !config.ssid.isEmpty else { return false }
    
    // Priorité dans la plage
    guard (0...100).contains(config.priority) else { return false }
    
    // Mot de passe suffisamment long si fourni
    if !config.password.isEmpty && config.password.count < 8 {
        showError("Le mot de passe doit faire au moins 8 caractères")
        return false
    }
    
    return true
}
```

## 🎯 Bonnes pratiques

### 1. Gestion de l'état

```swift
// Maintenir un état local synchronisé
@Published var wifiStatus: WiFiStatus?
@Published var configuredNetworks: [WiFiNetwork] = []

func refreshStatus() async {
    wifiStatus = try? await api.get("/wifi/status")
    configuredNetworks = try? await api.get("/wifi/networks").networks
}
```

### 2. Feedback utilisateur

```swift
// Toujours informer l'utilisateur de ce qui se passe
func performWiFiAction(_ action: () async throws -> Void) async {
    showLoading(true)
    
    do {
        try await action()
        showSuccess("Opération réussie")
    } catch {
        showError("Erreur: \(error.localizedDescription)")
    }
    
    showLoading(false)
    await refreshStatus()
}
```

### 3. Reconnexion automatique

```swift
// Si on perd la connexion au Pi
func handleConnectionLoss() {
    // 1. Vérifier si le hotspot est disponible
    if WiFi.availableNetworks.contains("NightScan-Setup") {
        showBanner("Pi en mode configuration", 
                  action: "Se connecter",
                  onTap: { WiFi.connect(to: "NightScan-Setup") })
    }
    
    // 2. Réessayer périodiquement
    retryTimer = Timer.scheduledTimer(withTimeInterval: 5.0) { _ in
        Task { await self.checkPiConnection() }
    }
}
```

## 🚨 Gestion des cas d'erreur

### Erreurs communes et solutions

| Erreur | Solution côté app |
|--------|-------------------|
| Pi non accessible | Proposer de se connecter au hotspot |
| Mot de passe incorrect | Redemander le mot de passe |
| Réseau non trouvé | Proposer un scan ou config manuelle |
| Quota dépassé | Impossible ici, pas de quota sur WiFi |
| Timeout | Augmenter le délai ou réessayer |

### Messages d'erreur clairs

```swift
enum WiFiError: LocalizedError {
    case piNotFound
    case wrongPassword
    case networkNotFound
    case connectionTimeout
    
    var errorDescription: String? {
        switch self {
        case .piNotFound:
            return "Impossible de trouver le Pi. Vérifiez qu'il est allumé et que le WiFi est activé (son)."
        case .wrongPassword:
            return "Mot de passe incorrect. Vérifiez et réessayez."
        case .networkNotFound:
            return "Réseau WiFi introuvable. Il est peut-être hors de portée."
        case .connectionTimeout:
            return "La connexion a pris trop de temps. Le Pi redémarre peut-être en mode hotspot."
        }
    }
}
```

## 🎉 Résumé

L'app iOS est maintenant le **centre de contrôle unique** pour toute la gestion WiFi :

- ✅ Configuration initiale simple
- ✅ Gestion complète des réseaux
- ✅ Pas besoin de navigateur web
- ✅ Expérience native iOS
- ✅ API REST complète et simple
- ✅ Feedback en temps réel

L'utilisateur n'a jamais besoin de quitter l'app pour configurer le WiFi ! 🚀