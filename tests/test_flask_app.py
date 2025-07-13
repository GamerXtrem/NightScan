import types
from pathlib import Path
import sys
import pytest
import io

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.helpers import ResponseAssertions, AuthenticationAssertions, FileUploadAssertions


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
    with client.session_transaction() as sess:
        sess["captcha_answer"] = "2"
    client.post(
        "/login",
        data={"username": "user", "password": "pass", "captcha": "2"},
    )

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
    
    # Enhanced assertions with comprehensive validation
    ResponseAssertions.assert_success_response(resp)
    
    # Validate file rejection with specific error message
    FileUploadAssertions.assert_upload_rejected(resp, ['invalid', 'wav', 'header'])
    assert b"Invalid WAV header" in resp.data
    
    # Verify that prediction service was not called due to invalid file
    assert not called, "Prediction service should not be called for invalid files"


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
    with client.session_transaction() as sess:
        sess["captcha_answer"] = "2"
    client.post(
        "/login",
        data={"username": "user", "password": "pass", "captcha": "2"},
    )

    data = {"file": (io.BytesIO(b"notriffdata"), "test.wav", "audio/x-wav")}
    resp = client.post("/", data=data, content_type="multipart/form-data")
    
    # Enhanced assertions with comprehensive validation
    ResponseAssertions.assert_success_response(resp)
    
    # Validate file rejection with specific error message
    FileUploadAssertions.assert_upload_rejected(resp, ['invalid', 'wav', 'header'])
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
    with client.session_transaction() as sess:
        sess["captcha_answer"] = "2"
    client.post(
        "/login",
        data={"username": "user", "password": "pass", "captcha": "2"},
    )

    resp = client.get("/")
    
    # Enhanced assertions with comprehensive validation
    ResponseAssertions.assert_success_response(resp)
    
    # Validate quota information is displayed
    assert b"Remaining quota" in resp.data, "Homepage should display remaining quota for logged in users"
    
    # Verify user is properly logged in
    AuthenticationAssertions.assert_user_logged_in(client, "user")


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
    
    # Enhanced assertions with comprehensive validation
    ResponseAssertions.assert_success_response(resp)
    
    # Validate password validation error message
    ResponseAssertions.assert_validation_error(resp, "password")
    assert b"Password must" in resp.data, "Should display password requirements message"


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
        with client.session_transaction() as sess:
            sess["captcha_answer"] = "2"
        client.post(
            "/login",
            data={"username": "user", "password": "wrong", "captcha": "2"},
        )

    with client.session_transaction() as sess:
        sess["captcha_answer"] = "2"
    resp = client.post(
        "/login",
        data={"username": "user", "password": "wrong", "captcha": "2"},
    )
    
    # Enhanced assertions with comprehensive validation
    ResponseAssertions.assert_rate_limit_error(resp)
    
    # Verify user is still not logged in after rate limiting
    AuthenticationAssertions.assert_user_logged_out(client)


def test_login_requires_captcha(monkeypatch, tmp_path):
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

    with client.session_transaction() as sess:
        sess["captcha_answer"] = "2"
    resp = client.post(
        "/login",
        data={"username": "user", "password": "pass1234", "captcha": "99"},
    )
    
    # Enhanced assertions for failed captcha validation
    ResponseAssertions.assert_success_response(resp)
    ResponseAssertions.assert_validation_error(resp, "captcha")
    assert b"Invalid captcha" in resp.data
    
    # Verify user is not logged in after failed captcha
    AuthenticationAssertions.assert_user_logged_out(client)

    # Test successful login with correct captcha
    with client.session_transaction() as sess:
        sess["captcha_answer"] = "2"
    resp = client.post(
        "/login",
        data={"username": "user", "password": "pass1234", "captcha": "2"},
    )
    
    # Enhanced assertions for successful login
    assert resp.status_code == 302, "Successful login should redirect"
    
    # Verify user is now logged in
    AuthenticationAssertions.assert_user_logged_in(client, "user")


def test_api_detections_returns_list(monkeypatch, tmp_path):
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
        det = module.Detection(
            species="Bat",
            latitude=1.0,
            longitude=2.0,
            zone="Z",
        )
        module.db.session.add_all([user, det])
        module.db.session.commit()

    client = app.test_client()
    client.environ_base["wsgi.url_scheme"] = "https"
    client.environ_base["HTTP_X_FORWARDED_PROTO"] = "https"
    with client.session_transaction() as sess:
        sess["captcha_answer"] = "2"
    client.post(
        "/login",
        data={"username": "user", "password": "pass", "captcha": "2"},
    )

    resp = client.get("/api/detections")
    
    # Enhanced assertions with comprehensive validation
    ResponseAssertions.assert_success_response(resp)
    
    # Validate API response structure
    data = resp.get_json()
    assert isinstance(data, list), "API detections should return a list"
    assert len(data) > 0, "Should have at least one detection"
    
    # Validate detection data structure
    detection = data[0]
    assert detection["species"] == "Bat", "Should return the created bat detection"
    
    # Validate detection has required fields
    required_fields = ['species', 'latitude', 'longitude', 'zone']
    for field in required_fields:
        assert field in detection, f"Detection should include '{field}' field"
        assert detection[field] is not None, f"Detection '{field}' should not be null"
    
    # Verify user has access to their own detections
    AuthenticationAssertions.assert_user_logged_in(client, "user")
