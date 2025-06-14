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
app.secret_key = "nightscan"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

PREDICT_API_URL = os.environ.get("PREDICT_API_URL", "http://localhost:8000/api/predict")


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
                )
                db.session.add(pred)
                db.session.commit()
            except requests.RequestException as e:
                flash(f"Prediction error: {e}")
    predictions = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.id.desc())
        .all()
    )
    return render_template("index.html", result=result, predictions=predictions)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=8000)
