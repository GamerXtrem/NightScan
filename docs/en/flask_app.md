# Flask Application

This document briefly explains how the web app in the `web/` folder works.

## Creating the SQLite database

When the app starts (`python web/app.py`), Flask creates a local SQLite database if it does not already exist:

```python
with app.app_context():
    db.create_all()
```

The `site.db` file lives in the same folder as the application. The `user` and `prediction` tables are generated from the models defined in `app.py`.

## Login and registration routes

Two pages let you create an account or log in:

- `GET /register` and `POST /register`: registration form asking for a user name and password. After successful creation, the user is prompted to log in.
- `GET /login` and `POST /login`: login form. After authentication the user is redirected to the index page.
- `GET /logout`: logs out the current session.

Access to the main page (`/`) is protected by `@login_required`: only logged‑in users can upload files and view their prediction history.

## Associating predictions with the user

Each result is stored in the `prediction` table with a `user_id` column. When an authenticated user submits a file:

```python
pred = Prediction(
    user_id=current_user.id,
    filename=file.filename,
    result=json.dumps(result),
)
```

The history displayed on the home page therefore only shows the predictions of the active user.

## Using another database

SQLite is suitable for testing or local use. To switch to MySQL (or any other SQLAlchemy‑supported backend), change the `SQLALCHEMY_DATABASE_URI` configuration in `web/app.py`:

```python
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://user:password@host/dbname"
```

Install the required connector (for example `pip install pymysql`) and run `db.create_all()` again within the application context to create the tables on the new database.

## Environment variables

Before starting the Flask server, define two variables:

- `SECRET_KEY`: used to sign the session. Choose a random value in production.
- `PREDICT_API_URL`: URL of the API that receives the files to analyze. If not set, `web/app.py` defaults to `http://localhost:8001/api/predict`. This variable may use either `http://` or `https://` depending on your API's configuration.

Example:

```bash
export SECRET_KEY="change-me"
export PREDICT_API_URL="http://myserver:8001/api/predict"
python web/app.py
```
