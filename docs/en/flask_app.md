# Flask Application

This document briefly explains how the web app in the `web/` folder works.

## Configuring the database

When the app starts (`python web/app.py`), Flask connects to the database specified by the `SQLALCHEMY_DATABASE_URI` environment variable. By default it uses MySQL:

```python
with app.app_context():
    db.create_all()
```

The `user` and `prediction` tables are generated from the models defined in `app.py`. If you prefer a local SQLite database for testing, set `SQLALCHEMY_DATABASE_URI` to something like `sqlite:///site.db` before running the app.

## Login and registration routes

Two pages let you create an account or log in:

- `GET /register` and `POST /register`: registration form asking for a user name and password. After successful creation, the user is prompted to log in.
- `GET /login` and `POST /login`: login form. After authentication the user is redirected to the index page.
- `GET /logout`: logs out the current session.

Access to the main page (`/`) is protected by `@login_required`: only logged‑in users can upload files and view their prediction history.

For public deployments you may instead send files from a WordPress site using
the upload plugin described in `docs/en/wordpress_plugin.md`. The built‑in form
is mainly for testing and can be omitted when uploads happen through
WordPress.

## Associating predictions with the user

Each result is stored in the `prediction` table with a `user_id` column. When an authenticated user submits a file:

```python
pred = Prediction(
    user_id=current_user.id,
    filename=file.filename,
    result=json.dumps(result),
    file_size=file_size,
)
```

`file_size` stores the uploaded file's size in bytes. The application rejects
files larger than 100 MB and prevents each user from uploading more than
10 GB in total.

The history displayed on the home page therefore only shows the predictions of the active user.

## Using another database

You may connect to any SQLAlchemy‑supported backend by adjusting the
`SQLALCHEMY_DATABASE_URI` configuration. After changing the URI, run
`db.create_all()` within the application context to create the tables on the new
database.

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
