# Guide Complet - Entraînement Audio sur Infomaniak Cloud

## 📋 Vue d'ensemble

Ce guide vous explique comment utiliser votre instance GPU Infomaniak Cloud pour entraîner rapidement le modèle audio NightScan. Votre instance `nvl4-a8-ram16-disk50-perf1` dispose d'une GPU NVIDIA L4 qui accélérera l'entraînement de 10-20x par rapport à un CPU.

### Spécifications de votre instance
- **Type**: nvl4-a8-ram16-disk50-perf1
- **GPU**: NVIDIA L4 (16GB VRAM)
- **CPU**: 8 vCPUs
- **RAM**: 16 GB
- **Stockage**: 50 GB SSD haute performance
- **IP**: 37.156.45.113

## 🚀 Étapes d'installation

### 1. Connexion à votre instance

```bash
ssh -i ~/.ssh/NightScan ubuntu@37.156.45.113
```

Si c'est votre première connexion, acceptez l'empreinte du serveur.

### 2. Setup automatique de l'environnement

Une fois connecté, exécutez le script de setup :

```bash
# Télécharger le script de setup
wget https://raw.githubusercontent.com/GamerXtrem/NightScan/main/scripts/setup_infomaniak_gpu.sh

# Le rendre exécutable et l'exécuter
chmod +x setup_infomaniak_gpu.sh
./setup_infomaniak_gpu.sh
```

Ce script va :
- Installer toutes les dépendances système
- Vérifier la présence de la GPU
- Cloner le projet NightScan
- Installer PyTorch avec support CUDA
- Créer la structure de répertoires
- Générer des scripts utilitaires

### 3. Transfert de votre dataset

#### Sur votre machine locale :

```bash
# Préparer le dataset pour le transfert
cd "/Users/jonathanmaitrot/NightScan/Clone claude/NightScan"
./scripts/prepare_dataset_transfer.sh
```

Le script va :
1. Compresser votre dataset audio
2. Vous proposer de le transférer automatiquement
3. Afficher les commandes pour l'extraction

#### Méthode manuelle (si nécessaire) :

```bash
# Compression du dataset
tar -czf dataset_audio.tar.gz "/Volumes/dataset/petit dataset_segmented"

# Transfert avec reprise possible
rsync -avzP -e "ssh -i ~/.ssh/NightScan" dataset_audio.tar.gz ubuntu@37.156.45.113:~/
```

### 4. Préparation du dataset sur l'instance

```bash
# Sur l'instance
cd ~/NightScan

# Extraire le dataset
tar -xzf ~/dataset_audio.tar.gz -C data/
mv "data/petit dataset_segmented" data/audio_data

# Vérifier l'extraction
find data/audio_data -type f -name "*.wav" | wc -l
```

### 5. Lancement de l'entraînement

#### Méthode recommandée avec tmux :

```bash
# Créer une session tmux (permet de se déconnecter)
tmux new -s training

# Dans la session tmux, lancer l'entraînement
~/NightScan/train_audio_cloud.sh data/audio_data 50 64
```

**Commandes tmux utiles :**
- Détacher : `Ctrl+B` puis `D`
- Réattacher : `tmux attach -t training`
- Lister sessions : `tmux ls`

#### Monitoring GPU (dans un autre terminal) :

```bash
ssh -i ~/.ssh/NightScan ubuntu@37.156.45.113
~/NightScan/monitor_gpu.sh
```

## 📊 Paramètres d'entraînement

### Configuration par défaut :
- **Epochs**: 50 (ajustable)
- **Batch size**: 64 (optimal pour L4 16GB)
- **Workers**: 4 (pour le chargement des données)
- **Mixed precision**: Activé automatiquement
- **Spectrogram caching**: Activé avec --pregenerate

### Temps d'entraînement estimés :
- **Sans GPU** : ~15 minutes/epoch = ~12h pour 50 epochs
- **Avec L4 GPU** : ~1-2 minutes/epoch = ~1-2h pour 50 epochs

### Ajustement des paramètres :

```bash
# Plus d'epochs pour meilleure précision
~/NightScan/train_audio_cloud.sh data/audio_data 100 64

# Batch size réduit si erreur mémoire
~/NightScan/train_audio_cloud.sh data/audio_data 50 32

# Test rapide
~/NightScan/train_audio_cloud.sh data/audio_data 5 16
```

## 💾 Récupération des résultats

### 1. Localiser le meilleur modèle :

```bash
# Sur l'instance
ls -lh ~/NightScan/audio_training_efficientnet/models/
```

Fichiers importants :
- `best_model.pth` : Le meilleur modèle
- `metadata.json` : Informations sur l'entraînement
- `training_history.json` : Métriques détaillées

### 2. Télécharger les résultats :

```bash
# Depuis votre machine locale
scp -i ~/.ssh/NightScan ubuntu@37.156.45.113:~/NightScan/audio_training_efficientnet/models/best_model.pth ./
scp -i ~/.ssh/NightScan ubuntu@37.156.45.113:~/NightScan/audio_training_efficientnet/models/*.json ./
```

## 🔧 Dépannage

### Problème : "CUDA out of memory"
```bash
# Réduire le batch size
~/NightScan/train_audio_cloud.sh data/audio_data 50 32
```

### Problème : GPU non détectée
```bash
# Vérifier les drivers
nvidia-smi
# Redémarrer si nécessaire
sudo reboot
```

### Problème : Entraînement interrompu
```bash
# Reprendre depuis le dernier checkpoint
tmux attach -t training
# Ou relancer (les checkpoints sont sauvegardés)
```

## 💰 Optimisation des coûts

1. **Arrêter l'instance après l'entraînement** :
   - Via l'interface Infomaniak
   - Ou créer un snapshot pour reprendre plus tard

2. **Utiliser les heures creuses** :
   - Les tarifs peuvent varier selon l'heure

3. **Batch multiple entraînements** :
   - Préparer plusieurs datasets
   - Les entraîner successivement

## 📝 Notes importantes

- **Sécurité** : Ne jamais partager votre clé SSH
- **Stockage** : L'instance a 50GB, surveillez l'espace
- **Persistence** : Les données sont perdues si l'instance est supprimée
- **Snapshots** : Créez des snapshots pour sauvegarder l'état

## 🎯 Workflow typique

1. **Préparation** (5-10 min)
   - Compression du dataset local
   - Transfert vers l'instance

2. **Setup** (10-15 min)
   - Exécution du script de setup
   - Extraction du dataset

3. **Entraînement** (1-2h)
   - Lancement dans tmux
   - Monitoring optionnel

4. **Récupération** (5 min)
   - Téléchargement des modèles
   - Nettoyage/arrêt de l'instance

**Temps total : ~2-3 heures** au lieu de 12+ heures sur CPU !

---

## Support

En cas de problème :
1. Vérifiez les logs dans `~/NightScan/logs/`
2. Consultez la sortie de `nvidia-smi`
3. Regardez l'utilisation mémoire avec `htop`