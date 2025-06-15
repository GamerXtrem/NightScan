# API de prédiction

Ce module fournit un petit serveur Flask exposant l'endpoint `POST /api/predict`. Il accepte un fichier WAV et renvoie les prédictions au format JSON comme celles affichées par `predict.py --json`.

## Lancer le serveur

Activez l'environnement virtuel puis lancez :

```bash
python Audio_Training/scripts/api_server.py \
  --model_path models/best_model.pth \
  --csv_dir data/processed/csv
```

Par défaut, l'API écoute sur `0.0.0.0:8001`. Les options `--host` et `--port` permettent de changer cette adresse.
Assurez-vous de ne pas utiliser le même port que l'application Flask afin d'éviter les collisions.
