import types
from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_app_module():
    path = Path(__file__).resolve().parents[1] / "web" / "app.py"
    source = path.read_text().replace("application = create_app()", "# application = create_app()")
    module = types.ModuleType("flask_app_test")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def test_create_app_requires_database_uri(monkeypatch):
    monkeypatch.delenv("SQLALCHEMY_DATABASE_URI", raising=False)
    monkeypatch.setenv("SECRET_KEY", "test")
    module = load_app_module()
    with pytest.raises(RuntimeError):
        module.create_app()
