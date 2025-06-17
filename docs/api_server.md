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

## Autoriser le domaine WordPress

Si l'API est appelée depuis un site tiers, le navigateur refusera la requête
sans en-tête CORS. Installez `flask_cors` puis ajoutez dans
`Audio_Training/scripts/api_server.py` :

```python
from flask_cors import CORS
CORS(app, origins=["https://mon-site-wordpress.exemple"])
```

Remplacez l'URL par celle de votre WordPress. L'entête
`Access-Control-Allow-Origin` contiendra ce domaine et l'envoi de fichiers
depuis le plugin fonctionnera.
