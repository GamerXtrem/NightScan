# 🚀 Guide de première installation WiFi NightScan

## 📱 Flux complet de configuration

### 1️⃣ **Première utilisation (aucun réseau configuré)**

1. **Réveil du Pi** 
   - Le Pi est en veille, WiFi désactivé
   - Faire un **son fort** près du Pi pour activer le WiFi
   - Le Pi détecte qu'aucun réseau n'est configuré

2. **Mode Hotspot automatique**
   - Le Pi démarre automatiquement en mode point d'accès
   - SSID : `NightScan-Setup`
   - Mot de passe : `nightscan2024`

3. **Configuration depuis l'app NightScan**
   - L'app détecte automatiquement le mode hotspot
   - Se connecter à `NightScan-Setup` depuis l'app
   - Scanner les réseaux disponibles dans l'app
   - Configurer vos réseaux WiFi directement dans l'app

4. **Connexion automatique**
   - Une fois configuré, le Pi se connecte au meilleur réseau
   - Le mode hotspot s'arrête automatiquement

### 2️⃣ **Utilisation quotidienne**

1. **Pi en veille** → Faire un son pour activer le WiFi
2. **Connexion automatique** au meilleur réseau disponible
3. **Si aucun réseau disponible** → Retour en mode hotspot

### 3️⃣ **Ajouter un nouveau réseau après installation**

**Option 1 : Depuis l'app NightScan (recommandé)**
- Aller dans Paramètres → WiFi
- Scanner les réseaux ou ajouter manuellement
- Configuration directe depuis l'app

**Option 2 : En mode hotspot**
- Si aucun réseau disponible, le Pi redémarre en hotspot
- L'app se reconnecte automatiquement à `NightScan-Setup`
- Configurer depuis l'app

**Option 3 : Via SSH (si accessible)**
```bash
sudo nightscan-wifi add "NouveauWiFi" "motdepasse"
```

## 🔄 Scénarios d'utilisation

### Scénario 1 : À la maison
1. **Son** → WiFi activé
2. Connexion automatique au WiFi maison (priorité élevée)
3. Si WiFi maison indisponible → Bascule sur hotspot iPhone
4. Si aucun réseau → Mode hotspot pour reconfiguration

### Scénario 2 : Sur le terrain
1. **Son** → WiFi activé
2. Connexion au hotspot iPhone
3. Si pas d'iPhone → Mode hotspot pour utiliser avec l'app

### Scénario 3 : Nouveau lieu
1. **Son** → WiFi activé
2. Aucun réseau connu → Mode hotspot
3. Configuration du nouveau WiFi
4. Connexion automatique

## 📊 Priorités recommandées

| Réseau | Priorité | Usage |
|--------|----------|--------|
| WiFi Maison | 100 | Priorité maximale à la maison |
| WiFi Bureau | 90 | Lieu de travail principal |
| Hotspot iPhone | 70 | Usage mobile principal |
| WiFi Public | 30 | Réseaux occasionnels |
| WiFi Invité | 10 | Réseaux temporaires |

## 🛠️ Configuration avancée

### Modifier les priorités
Via l'interface web, section "Options avancées" :
- **Priorité** : 0-100 (plus élevé = préféré)
- **Connexion auto** : Activer/Désactiver
- **Réseau caché** : Pour les SSID non diffusés
- **Notes** : Aide-mémoire personnel

### Paramètres du hotspot
Le hotspot peut être personnalisé :
```json
{
  "ssid": "MonNightScan",
  "password": "MonMotDePasse2024",
  "channel": 6,
  "hidden": false
}
```

## ❓ FAQ

**Q : Dois-je faire un son à chaque utilisation ?**
R : Oui, c'est le mécanisme d'économie d'énergie. Le WiFi reste actif 10 minutes après activation.

**Q : Comment savoir si le Pi est en mode hotspot ?**
R : Recherchez le réseau `NightScan-Setup` dans les réglages WiFi de l'iPhone.

**Q : Puis-je avoir plusieurs réseaux avec la même priorité ?**
R : Oui, le Pi choisira celui avec le meilleur signal.

**Q : Le Pi peut-il mémoriser des réseaux 5GHz ?**
R : Cela dépend du modèle de Pi. Pi 3B+ et 4 supportent le 5GHz.

**Q : Combien de réseaux puis-je configurer ?**
R : Aucune limite, mais trop de réseaux peut ralentir la connexion initiale.

## 🚨 Dépannage

### Le hotspot n'apparaît pas
1. Vérifier que le son a bien activé le WiFi
2. Attendre 30 secondes après le son
3. Rafraîchir la liste des réseaux sur l'iPhone

### Impossible de se connecter au hotspot
1. Vérifier le mot de passe : `nightscan2024`
2. Oublier le réseau et réessayer
3. Redémarrer le Pi si nécessaire

### L'app ne trouve pas le Pi
1. Vérifier la connexion au hotspot `NightScan-Setup`
2. Redémarrer l'app NightScan
3. Attendre que l'app détecte automatiquement le mode hotspot

### Le Pi ne se connecte pas après configuration
1. Vérifier le mot de passe saisi dans l'app
2. Vérifier que le réseau est à portée
3. Utiliser la configuration manuelle dans l'app si réseau caché

## 💡 Astuces

1. **Prioriser intelligemment** : WiFi fixe > Hotspot mobile > Public
2. **Notes utiles** : Ajouter le lieu dans les notes
3. **Test de portée** : Configurer en étant proche du routeur
4. **Sauvegarde** : Noter les réseaux configurés
5. **Sécurité** : Changer le mot de passe du hotspot après installation

---

Le système est conçu pour être simple et automatique. Une fois configuré, vous n'avez qu'à faire un son pour activer le Pi ! 🎉