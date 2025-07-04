# Flask Application

This document briefly explains how the web app in the `web/` folder works.

## Configuring the database

When the app starts (`python web/app.py`) it expects the `SQLALCHEMY_DATABASE_URI` environment variable to be defined. `create_app()` raises `RuntimeError` if it is missing. Use a URI containing secure credentials appropriate for your database backend:

```python
with app.app_context():
    db.create_all()
```

The `user` and `prediction` tables are generated from the models defined in `app.py`. If you prefer a local SQLite database for testing, set `SQLALCHEMY_DATABASE_URI` to something like `sqlite:///site.db` before running the app.
For public deployments supply secure credentials in `SQLALCHEMY_DATABASE_URI` so the database cannot be accessed with default passwords.

## Login and registration routes

Two pages let you create an account or log in:

- `GET /register` and `POST /register`: registration form asking for a user name and password. After successful creation, the user is prompted to log in.
- `GET /login` and `POST /login`: login form. After authentication the user is redirected to the index page.
- `GET /logout`: logs out the current session.

Access to the main page (`/`) is protected by `@login_required`: only logged‑in users can upload files and view their prediction history.

### Authentication security

Passwords must be at least **10 characters** and contain a lowercase letter, uppercase letter, digit and symbol. Login requests are rate limited to **5 per minute** and a temporary lockout is triggered after five failed attempts from the same IP address within 30 minutes.

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

`file_size` stores the uploaded file's size in bytes.

### Upload limits

The application rejects files larger than 100 MB and prevents each user from uploading more than 10 GB in total.

The history displayed on the home page therefore only shows the predictions of the active user.

## Using another database

You may connect to any SQLAlchemy‑supported backend by adjusting the
`SQLALCHEMY_DATABASE_URI` configuration. After changing the URI, run
`db.create_all()` within the application context to create the tables on the new
database.

## Environment variables

Before starting the Flask server, define two variables:

- `SECRET_KEY`: used to sign the session. If unset, the application generates a temporary value at startup. Provide a persistent random key in production.
- `PREDICT_API_URL`: URL of the API that receives the files to analyze. If not set, `web/app.py` defaults to `http://localhost:8001/api/predict`. The application accepts either scheme, but in production use an `https://` endpoint.

Example (install `gunicorn` if it is not already available):

```bash
pip install gunicorn
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
export PREDICT_API_URL="https://myserver.example/api/predict"
gunicorn -w 4 -b 0.0.0.0:8000 web.app:application
```

Start a Celery worker to handle the prediction requests asynchronously:

```bash
celery -A web.tasks worker --loglevel=info
```

In production you should place a reverse proxy such as Nginx in front of the
Gunicorn workers and forward requests to port `8000`.

The command above binds the server to `0.0.0.0`, which makes the application
reachable from any network interface. When you deploy behind a reverse proxy or
have a firewall restricting access, this is normally fine. If you want the
service to be accessible only locally, change the host to `127.0.0.1` or add
appropriate firewall rules. When the app is accessible on the public Internet
you should enable HTTPS so login credentials and uploads are protected in
transit.
