import types
from pathlib import Path
import sys
import pytest
import io

sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_app_module():
    path = Path(__file__).resolve().parents[1] / "web" / "app.py"
    source = path.read_text().replace("application = create_app()", "# application = create_app()")
    module = types.ModuleType("web.app")
    module.__file__ = str(path)
    module.__package__ = "web"
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def test_create_app_requires_database_uri(monkeypatch):
    monkeypatch.delenv("SQLALCHEMY_DATABASE_URI", raising=False)
    monkeypatch.setenv("SECRET_KEY", "test")
    module = load_app_module()
    with pytest.raises(RuntimeError):
        module.create_app()


def test_index_rejects_bad_mimetype(monkeypatch, tmp_path):
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv(
        "SQLALCHEMY_DATABASE_URI", f"sqlite:///{tmp_path / 'db.sqlite'}"
    )
    module = load_app_module()
    app = module.create_app()
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        user = module.User(username="user")
        user.set_password("pass")
        module.db.session.add(user)
        module.db.session.commit()

    client = app.test_client()
    client.environ_base["wsgi.url_scheme"] = "https"
    client.environ_base["HTTP_X_FORWARDED_PROTO"] = "https"
    client.post("/login", data={"username": "user", "password": "pass"})

    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True

        class Dummy:
            def raise_for_status(self):
                pass

            def json(self):
                return {}

        return Dummy()

    monkeypatch.setattr(module.requests, "post", fake_post)
    data = {"file": (io.BytesIO(b"data"), "test.wav", "text/plain")}
    resp = client.post("/", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Invalid WAV header" in resp.data
    assert not called


def test_index_rejects_bad_header(monkeypatch, tmp_path):
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv(
        "SQLALCHEMY_DATABASE_URI", f"sqlite:///{tmp_path / 'db.sqlite'}"
    )
    module = load_app_module()
    app = module.create_app()
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        user = module.User(username="user")
        user.set_password("pass")
        module.db.session.add(user)
        module.db.session.commit()

    client = app.test_client()
    client.environ_base["wsgi.url_scheme"] = "https"
    client.environ_base["HTTP_X_FORWARDED_PROTO"] = "https"
    client.post("/login", data={"username": "user", "password": "pass"})

    data = {"file": (io.BytesIO(b"notriffdata"), "test.wav", "audio/x-wav")}
    resp = client.post("/", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Invalid WAV header" in resp.data


def test_index_shows_remaining_quota(monkeypatch, tmp_path):
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv(
        "SQLALCHEMY_DATABASE_URI", f"sqlite:///{tmp_path / 'db.sqlite'}"
    )
    module = load_app_module()
    app = module.create_app()
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        user = module.User(username="user")
        user.set_password("pass")
        module.db.session.add(user)
        module.db.session.commit()

    client = app.test_client()
    client.environ_base["wsgi.url_scheme"] = "https"
    client.environ_base["HTTP_X_FORWARDED_PROTO"] = "https"
    client.post("/login", data={"username": "user", "password": "pass"})

    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Remaining quota" in resp.data


def test_register_password_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv(
        "SQLALCHEMY_DATABASE_URI", f"sqlite:///{tmp_path / 'db.sqlite'}"
    )
    module = load_app_module()
    app = module.create_app()
    app.config["WTF_CSRF_ENABLED"] = False
    client = app.test_client()
    client.environ_base["wsgi.url_scheme"] = "https"
    client.environ_base["HTTP_X_FORWARDED_PROTO"] = "https"

    resp = client.post("/register", data={"username": "u", "password": "short"})
    assert resp.status_code == 200
    assert b"Password must" in resp.data


def test_login_rate_limiting(monkeypatch, tmp_path):
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv(
        "SQLALCHEMY_DATABASE_URI", f"sqlite:///{tmp_path / 'db.sqlite'}"
    )
    module = load_app_module()
    app = module.create_app()
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        user = module.User(username="user")
        user.set_password("pass1234")
        module.db.session.add(user)
        module.db.session.commit()

    client = app.test_client()
    client.environ_base["wsgi.url_scheme"] = "https"
    client.environ_base["HTTP_X_FORWARDED_PROTO"] = "https"

    for _ in range(5):
        client.post("/login", data={"username": "user", "password": "wrong"})

    resp = client.post("/login", data={"username": "user", "password": "wrong"})
    assert resp.status_code == 429
