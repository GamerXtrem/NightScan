# Application Flask

Ce document décrit brièvement le fonctionnement de l'application Web située dans le dossier `web/`.

## Création de la base SQLite

Lors du démarrage de l'application (`python web/app.py`), Flask crée une base SQLite locale si elle n'existe pas déjà. Cela se fait par l'appel suivant :

```python
with app.app_context():
    db.create_all()
```

La base `site.db` est stockée dans le même dossier que l'application. Les tables `user` et `prediction` sont générées à partir des modèles définis dans `app.py`.

## Routes de connexion et d'inscription

Deux pages permettent d'enregistrer un nouvel utilisateur ou de se connecter :

- `GET /register` et `POST /register` : formulaire d'inscription. Un nom d'utilisateur et un mot de passe sont demandés. Une fois la création réussie, l'utilisateur est invité à se connecter.
- `GET /login` et `POST /login` : formulaire de connexion. Après authentification, l'utilisateur est redirigé vers l'index.
- `GET /logout` : déconnexion de la session en cours.

L'accès à la page principale (`/`) est protégé par `@login_required` : seul un utilisateur connecté peut envoyer des fichiers et consulter son historique de prédictions.

Pour un déploiement public, vous pouvez aussi transmettre les fichiers depuis
un site WordPress grâce au plugin d'envoi décrit dans
`docs/wordpress_plugin.md`. Le formulaire local reste utile pour les tests mais
n'est pas obligatoire si l'envoi se fait depuis WordPress.

## Association des prédictions à l'utilisateur

Chaque résultat est enregistré dans la table `prediction` avec la colonne `user_id`. Lorsqu'un utilisateur connecté soumet un fichier :

```python
pred = Prediction(
    user_id=current_user.id,
    filename=file.filename,
    result=json.dumps(result),
)
```

Ainsi, l'historique affiché sur la page d'accueil correspond uniquement aux prédictions de l'utilisateur actif.

## Utiliser une autre base de données

SQLite convient pour les tests ou un usage local. Pour passer sur MySQL (ou tout autre SGBD pris en charge par SQLAlchemy), modifiez la variable de configuration `SQLALCHEMY_DATABASE_URI` dans `web/app.py` :

```python
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://nom:motdepasse@hote/basededonnees"
```

Installez ensuite le connecteur nécessaire (par exemple `pip install pymysql`) et exécutez à nouveau `db.create_all()` dans le contexte de l'application pour créer les tables sur cette nouvelle base.

## Variables d'environnement

Avant de démarrer le serveur Flask, deux variables doivent être définies :

- `SECRET_KEY` : chaîne servant à signer la session. Choisissez une valeur
  aléatoire pour un déploiement réel.
- `PREDICT_API_URL` : URL de l'API recevant les fichiers à analyser. Sans
  configuration explicite, `web/app.py` se rabat sur
  `http://localhost:8001/api/predict`. Cette variable peut pointer vers un
  endpoint en `http://` ou `https://` selon votre configuration.

Exemple :

```bash
export SECRET_KEY="change-me"
export PREDICT_API_URL="http://monserveur:8001/api/predict"
python web/app.py
```

Avant de lancer l'application Web, assurez-vous que l'API de prédiction est
active. Elle se démarre avec :

```bash
python Audio_Training/scripts/api_server.py \
  --model_path models/best_model.pth \
  --csv_dir data/processed/csv
```

Par défaut, l'API écoute sur `0.0.0.0:8001`. Vérifiez que `PREDICT_API_URL`
utilise bien ce port ou ajustez-le si nécessaire pour éviter toute collision
avec l'application Flask.
