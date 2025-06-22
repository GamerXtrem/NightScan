import os
import json
import requests

from flask import (
    Flask,
    request,
    render_template,
    flash,
    redirect,
    url_for,
)
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user,
    UserMixin,
)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    raise RuntimeError("SECRET_KEY environment variable not set")
app.secret_key = secret_key
app.config["WTF_CSRF_SECRET_KEY"] = os.environ.get("WTF_CSRF_SECRET_KEY", secret_key)
csrf = CSRFProtect(app)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy()

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB per upload
MAX_TOTAL_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB per user

login_manager = LoginManager(app)
login_manager.login_view = "login"

PREDICT_API_URL = os.environ.get("PREDICT_API_URL", "http://localhost:8001/api/predict")


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    filename = db.Column(db.String(200))
    result = db.Column(db.Text)
    file_size = db.Column(db.Integer)

    user = db.relationship("User", backref=db.backref("predictions", lazy=True))


@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            flash("Please provide username and password")
        elif User.query.filter_by(username=username).first():
            flash("Username already exists")
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Registration successful. Please log in.")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("index"))
        flash("Invalid credentials")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.lower().endswith(".wav"):
            flash("Please upload a WAV file.")
        elif file.mimetype not in ("audio/wav", "audio/x-wav"):
            flash("Invalid file type. Only WAV files are accepted.")
        else:
            file_size = request.content_length
            if file_size is None:
                pos = file.stream.tell()
                file.stream.seek(0, os.SEEK_END)
                file_size = file.stream.tell()
                file.stream.seek(pos)
            if file_size > MAX_FILE_SIZE:
                flash("File exceeds 100 MB limit.")
            else:
                total = (
                    db.session.query(
                        db.func.coalesce(db.func.sum(Prediction.file_size), 0)
                    )
                    .filter_by(user_id=current_user.id)
                    .scalar()
                    or 0
                )
                if total + file_size > MAX_TOTAL_SIZE:
                    flash("Upload quota exceeded (10 GB total).")
                else:
                    try:
                        resp = requests.post(
                            PREDICT_API_URL,
                            files={"file": (file.filename, file.stream, "audio/wav")},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        pred = Prediction(
                            user_id=current_user.id,
                            filename=file.filename,
                            result=json.dumps(result),
                            file_size=file_size,
                        )
                        db.session.add(pred)
                        db.session.commit()
                    except requests.RequestException as e:
                        app.logger.error("Prediction request failed: %s", e)
                        flash("Prediction failed. Please try again later.")
    predictions = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.id.desc())
        .all()
    )
    return render_template("index.html", result=result, predictions=predictions)


def create_app():
    """Initialize the database and return the Flask app."""
    db_uri = os.environ.get("SQLALCHEMY_DATABASE_URI")
    if not db_uri:
        raise RuntimeError("SQLALCHEMY_DATABASE_URI environment variable not set")
    app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app


application = create_app()


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8000)
