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
