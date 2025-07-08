# 📶 NightScan WiFi Management System

## Vue d'ensemble

Le système WiFi de NightScan permet au Raspberry Pi de se connecter automatiquement à plusieurs réseaux WiFi avec un système de priorités et un basculement automatique vers un mode hotspot pour la configuration.

## 🎯 Fonctionnalités principales

### ✅ Gestion multi-réseaux
- **Stockage de plusieurs réseaux** avec priorités
- **Connexion automatique** au meilleur réseau disponible
- **Basculement intelligent** entre réseaux
- **Persistance des configurations** en JSON

### ✅ Mode hotspot automatique
- **Point d'accès automatique** si aucun réseau disponible
- **Configuration via app iOS** native et intuitive
- **Configuration hostapd/dnsmasq** automatique
- **Retour automatique** en mode client quand réseau disponible

### ✅ API REST complète
- **Endpoints pour gestion des réseaux** (CRUD)
- **Scan des réseaux WiFi** disponibles
- **Connexion/déconnexion** programmatique
- **Status et monitoring** en temps réel

### ✅ Monitoring et récupération
- **Surveillance automatique** des connexions
- **Reconnexion automatique** en cas de perte
- **Logs détaillés** pour diagnostic
- **Service systemd** pour démarrage automatique

## 📁 Structure des fichiers

```
NightScanPi/Program/
├── wifi_manager.py              # Gestionnaire WiFi principal
├── wifi_setup_web.py            # Interface web de configuration
├── wifi_service.py              # API REST (mise à jour)
├── wifi_config.py               # Configuration legacy (maintenu)
└── scripts/
    ├── setup_hotspot.sh         # Configuration système hotspot
    ├── install_wifi_service.sh  # Installation du service
    └── wifi_startup.py          # Service de démarrage
```

## 🚀 Installation

### 1. Configuration système

```bash
# Installer les dépendances
sudo apt update
sudo apt install hostapd dnsmasq iptables-persistent

# Configurer le système pour le hotspot
sudo ./scripts/setup_hotspot.sh

# Installer le service systemd
sudo ./scripts/install_wifi_service.sh
```

### 2. Configuration des réseaux

```bash
# Ajouter un réseau WiFi
sudo nightscan-wifi add "MonWiFi" "motdepasse123"

# Scanner les réseaux disponibles
sudo nightscan-wifi scan

# Se connecter à un réseau
sudo nightscan-wifi connect "MonWiFi"

# Voir le status
sudo nightscan-wifi status
```

### 3. Démarrage du service

```bash
# Démarrer le service
sudo systemctl start nightscan-wifi

# Activer au démarrage
sudo systemctl enable nightscan-wifi

# Voir les logs
sudo journalctl -u nightscan-wifi -f
```

## 🌐 API REST

Le système expose une API REST complète pour la gestion des réseaux WiFi :

### Gestion des réseaux

```bash
# Lister les réseaux configurés
GET /wifi/networks

# Ajouter un réseau
POST /wifi/networks
{
  "ssid": "MonWiFi",
  "password": "motdepasse",
  "priority": 50,
  "auto_connect": true,
  "hidden": false,
  "notes": "WiFi de la maison"
}

# Supprimer un réseau
DELETE /wifi/networks/MonWiFi

# Modifier un réseau
PUT /wifi/networks/MonWiFi
{
  "priority": 100,
  "notes": "Priorité élevée"
}
```

### Connexion et statut

```bash
# Scanner les réseaux
GET /wifi/scan

# Se connecter à un réseau
POST /wifi/connect/MonWiFi
{
  "force": true
}

# Se connecter au meilleur réseau
POST /wifi/connect/best

# Se déconnecter
POST /wifi/disconnect

# Statut détaillé
GET /wifi/status
```

### Hotspot

```bash
# Démarrer le hotspot
POST /wifi/hotspot/start

# Arrêter le hotspot
POST /wifi/hotspot/stop

# Configuration du hotspot
GET /wifi/hotspot/config
PUT /wifi/hotspot/config
{
  "ssid": "NightScan-Setup",
  "password": "nouveaumotdepasse",
  "channel": 6
}
```

## 📱 Configuration via l'app iOS

L'app NightScan est le seul point de configuration WiFi :

### Fonctionnalités
- **Détection automatique** du mode hotspot
- **Scan des réseaux WiFi** intégré dans l'app
- **Configuration des réseaux** avec interface native
- **Gestion des priorités** intuitive
- **API REST** complète pour toutes les opérations

### Utilisation
1. **L'app détecte** automatiquement le mode hotspot du Pi
2. **Se connecter** au hotspot `NightScan-Setup` via l'app
3. **Scanner et configurer** directement dans l'interface native
4. **Connexion automatique** - le Pi bascule en mode client
5. **Gestion continue** des réseaux depuis les paramètres de l'app

## 🔧 Configuration avancée

### Fichiers de configuration

```bash
# Réseaux WiFi
/opt/nightscan/config/wifi_networks.json

# Configuration hotspot
/opt/nightscan/config/hotspot_config.json

# Statut système
/opt/nightscan/config/wifi_status.json
```

### Exemple de configuration réseau

```json
{
  "WiFi_Maison": {
    "ssid": "WiFi_Maison",
    "password": "motdepasse123",
    "security": "wpa2_psk",
    "priority": 100,
    "auto_connect": true,
    "hidden": false,
    "country": "FR",
    "last_connected": 1704067200.0,
    "connection_attempts": 0,
    "notes": "WiFi principal de la maison"
  },
  "Hotspot_Mobile": {
    "ssid": "Hotspot_Mobile",
    "password": "mobile123",
    "security": "wpa2_psk",
    "priority": 50,
    "auto_connect": true,
    "hidden": false,
    "notes": "Hotspot de l'iPhone"
  }
}
```

### Configuration hotspot

```json
{
  "ssid": "NightScan-Setup",
  "password": "nightscan2024",
  "channel": 6,
  "hidden": false,
  "max_clients": 10,
  "ip_range": "192.168.4.0/24",
  "gateway": "192.168.4.1",
  "dhcp_start": "192.168.4.100",
  "dhcp_end": "192.168.4.200"
}
```

## 🔄 Fonctionnement automatique

### Algorithme de connexion

1. **Démarrage** : Le service démarre automatiquement
2. **Scan** : Recherche des réseaux disponibles
3. **Priorité** : Sélection du réseau avec la priorité la plus élevée
4. **Connexion** : Tentative de connexion au réseau sélectionné
5. **Fallback** : Si échec, tentative avec le réseau suivant
6. **Hotspot** : Si aucun réseau disponible, démarrage du hotspot

### Surveillance continue

- **Vérification** toutes les 30 secondes
- **Reconnexion automatique** en cas de perte
- **Basculement** vers un autre réseau si nécessaire
- **Logs détaillés** pour diagnostic

## 📊 Monitoring et logs

### Logs système

```bash
# Logs du service principal
sudo journalctl -u nightscan-wifi -f

# Logs de démarrage
sudo tail -f /opt/nightscan/logs/wifi_startup.log

# Logs du gestionnaire WiFi
sudo tail -f /opt/nightscan/logs/wifi_manager.log
```

### Statut en temps réel

```bash
# Statut détaillé
sudo nightscan-wifi status

# Réseaux configurés
sudo nightscan-wifi networks

# Scan des réseaux
sudo nightscan-wifi scan
```

## 🛠️ Résolution de problèmes

### Problèmes courants

#### 1. Interface wlan0 non disponible
```bash
# Vérifier l'interface
ip link show wlan0

# Activer l'interface
sudo ip link set wlan0 up
```

#### 2. Hotspot ne démarre pas
```bash
# Vérifier hostapd
sudo systemctl status hostapd

# Tester la configuration
sudo hostapd -d /etc/hostapd/hostapd.conf
```

#### 3. Pas de connexion automatique
```bash
# Vérifier le service
sudo systemctl status nightscan-wifi

# Redémarrer le service
sudo systemctl restart nightscan-wifi
```

#### 4. Configuration corrompue
```bash
# Réinitialiser la configuration
sudo rm /opt/nightscan/config/wifi_networks.json
sudo systemctl restart nightscan-wifi
```

### Diagnostic avancé

```bash
# Test de connectivité
ping -c 3 8.8.8.8

# Statut des interfaces
ip addr show

# Processus WiFi
ps aux | grep -E "(wpa_supplicant|hostapd|dnsmasq)"

# Configuration wpa_supplicant
sudo cat /etc/wpa_supplicant/wpa_supplicant.conf
```

## 🔒 Sécurité

### Bonnes pratiques

1. **Mots de passe forts** pour le hotspot
2. **Chiffrement WPA2** pour tous les réseaux
3. **Accès restreint** au mode hotspot
4. **Logs sécurisés** avec rotation automatique
5. **Permissions système** appropriées

### Configuration sécurisée

```json
{
  "ssid": "NightScan-Secure",
  "password": "MotDePasseComplexe123!",
  "channel": 6,
  "hidden": true,
  "max_clients": 5
}
```

## 📈 Cas d'usage

### 1. Utilisation domestique
- **WiFi maison** (priorité 100)
- **Hotspot mobile** (priorité 50)
- **Invités** (priorité 10)

### 2. Utilisation professionnelle
- **Réseau bureau** (priorité 100)
- **WiFi portable** (priorité 70)
- **Hotspot secours** (priorité 30)

### 3. Utilisation sur le terrain
- **Hotspot iPhone** (priorité 80)
- **WiFi public** (priorité 40)
- **Mode AP** pour configuration

## 🚀 Intégration avec l'app iOS

### Fonctionnalités disponibles

1. **Scan et connexion** depuis l'app
2. **Gestion des réseaux** sauvegardés
3. **Configuration du hotspot** personnalisé
4. **Monitoring en temps réel**
5. **Diagnostic à distance**

### Endpoints pour l'app

```python
# Dans votre app iOS
let networks = await api.get("/wifi/networks")
let success = await api.post("/wifi/connect/\(ssid)")
let status = await api.get("/wifi/status")
```

## 🔮 Améliorations futures

### Fonctionnalités planifiées

1. **Géolocalisation** des réseaux
2. **Profils de configuration** par lieu
3. **Statistiques d'utilisation** détaillées
4. **Notifications push** pour l'app
5. **Synchronisation cloud** des configurations

### Optimisations possibles

1. **Cache des réseaux** pour performance
2. **Prédiction de connexion** basée sur l'historique
3. **Gestion d'énergie** avancée
4. **Équilibrage de charge** entre réseaux

---

## 📞 Support

Pour plus d'informations ou en cas de problème :

1. **Logs détaillés** : `sudo journalctl -u nightscan-wifi -f`
2. **Status système** : `sudo nightscan-wifi status`
3. **Configuration** : `/opt/nightscan/config/`
4. **Documentation** : Ce fichier README

Le système WiFi de NightScan est maintenant prêt pour un usage en production avec une gestion automatique complète et une interface intuitive ! 🎉