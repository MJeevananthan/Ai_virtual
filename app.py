"""
AI Virtual Doctor — Flask App
MySQL via SQLAlchemy (local + Railway/PlanetScale cloud)
"""
import os, pickle, ast
import numpy as np
import pandas as pd
import bcrypt
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash)
from flask_sqlalchemy import SQLAlchemy
from groq import Groq
from datetime import datetime

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
except Exception:
    pass

# ── App ────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.secret_key = os.environ.get("SECRET_KEY", "aivdoctor_secret_2026")

# ── Database (MySQL always) ────────────────────────────────────────────────────
MYSQL_USER = os.environ.get("MYSQL_USER",     "root")
MYSQL_PASS = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_HOST = os.environ.get("MYSQL_HOST",     "localhost")
MYSQL_PORT = os.environ.get("MYSQL_PORT",     "3306")
MYSQL_DB   = os.environ.get("MYSQL_DB",       "ai_doctor")

# Render / Railway gives MYSQL_URL directly — use if available
MYSQL_URL  = os.environ.get("MYSQL_URL", "")
if not MYSQL_URL:
    MYSQL_URL = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    )

app.config["SQLALCHEMY_DATABASE_URI"]        = MYSQL_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"]      = {
    "pool_recycle": 280,
    "pool_pre_ping": True,
}

db = SQLAlchemy(app)

# ── Models ─────────────────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = "users"
    id         = db.Column(db.Integer, primary_key=True)
    full_name  = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(150), unique=True, nullable=False)
    password   = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatHistory(db.Model):
    __tablename__ = "chat_history"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role       = db.Column(db.String(20), nullable=False)
    message    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    __tablename__ = "predictions"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    symptoms   = db.Column(db.Text, nullable=False)
    predicted  = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    try:
        db.create_all()
        print("[OK] MySQL tables ready")
    except Exception as e:
        print(f"[WARN] DB error: {e}")

# ── Groq LLM ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── Load ML Model ─────────────────────────────────────────────────────────────
def load_artifacts():
    model_path = os.path.join(BASE_DIR, "model", "disease_model.pkl")
    enc_path   = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
    symp_path  = os.path.join(BASE_DIR, "model", "symptoms_list.pkl")
    with open(model_path, "rb") as f: model = pickle.load(f)
    with open(enc_path,   "rb") as f: le    = pickle.load(f)
    with open(symp_path,  "rb") as f: syms  = pickle.load(f)
    return model, le, syms

model, le, SYMPTOMS = load_artifacts()

SYM_NORM_MAP = {}
for s in SYMPTOMS:
    SYM_NORM_MAP[s.strip().lower()]                   = s
    SYM_NORM_MAP[s.strip().lower().replace("_", " ")] = s

print(f"[OK] Model loaded | Symptoms: {len(SYMPTOMS)} | Classes: {len(le.classes_)}")

# ── CSV info maps ──────────────────────────────────────────────────────────────
def _csv(name):
    return pd.read_csv(os.path.join(BASE_DIR, "dataset", name))

desc_df = _csv("description.csv")
diet_df = _csv("diets.csv")
med_df  = _csv("medications.csv")
prec_df = _csv("precautions.csv")
work_df = _csv("workout.csv")

def _norm(s): return str(s).strip().lower()

desc_map = {_norm(r["Disease"]): str(r["Description"]) for _, r in desc_df.iterrows()}
diet_map = {_norm(r["Disease"]): r["Diet"]             for _, r in diet_df.iterrows()}
med_map  = {_norm(r["Disease"]): r["Medication"]       for _, r in med_df.iterrows()}
work_map = {_norm(r["Disease"]): r["Workouts"]         for _, r in work_df.iterrows()}
prec_map = {}
for _, r in prec_df.iterrows():
    key = _norm(r["Disease"])
    prec_map[key] = [str(r.get(f"Precaution_{i}", "")).strip()
                     for i in range(1, 5)
                     if str(r.get(f"Precaution_{i}", "")).strip() not in ("", "nan")]

def _parse(raw):
    if isinstance(raw, list): return raw
    try:
        v = ast.literal_eval(str(raw))
        if isinstance(v, list): return [str(x).strip() for x in v]
    except: pass
    return [x.strip() for x in str(raw).split(",") if x.strip() and x.strip() != "nan"]

def get_info(disease):
    k = _norm(disease)
    return {
        "description": desc_map.get(k, "Description not available."),
        "diet":        _parse(diet_map.get(k, [])),
        "medications": _parse(med_map.get(k, [])),
        "precautions": prec_map.get(k, []),
        "workouts":    _parse(work_map.get(k, [])),
    }

# ── Auth decorator ─────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def root():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        user     = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode(), user.password.encode()):
            session["user_id"]   = user.id
            session["user_name"] = user.full_name
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name     = request.form.get("full_name", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if not all([name, email, password]):
            flash("All fields are required.", "error")
            return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return render_template("register.html")
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        db.session.add(User(full_name=name, email=email, password=hashed))
        db.session.commit()
        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    display_syms = [s.replace("_", " ") for s in SYMPTOMS]
    return render_template("index.html",
                           symptoms=display_syms,
                           user_name=session.get("user_name", ""))

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION API
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/symptoms")
@login_required
def api_symptoms():
    return jsonify({"symptoms": [s.replace("_", " ") for s in SYMPTOMS]})

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(force=True)
    if not data or "symptoms" not in data:
        return jsonify({"error": "No symptoms provided"}), 400
    raw = [s.strip().lower() for s in data["symptoms"] if s.strip()]
    if not raw:
        return jsonify({"error": "Symptom list is empty"}), 400

    input_vec    = np.zeros(len(SYMPTOMS), dtype=np.float32)
    unrecognized = []
    matched      = []
    for sym in raw:
        col = SYM_NORM_MAP.get(sym) or SYM_NORM_MAP.get(sym.replace(" ", "_"))
        if col:
            input_vec[SYMPTOMS.index(col)] = 1.0
            matched.append(col)
        else:
            unrecognized.append(sym)

    if input_vec.sum() == 0:
        return jsonify({"error": f"Symptoms not recognized: {unrecognized}"}), 400

    proba     = model.predict_proba([input_vec])[0]
    top_idx   = np.argsort(proba)[::-1][:3]
    top_preds = [{"disease": le.inverse_transform([i])[0],
                  "confidence": round(float(proba[i]) * 100, 2)}
                 for i in top_idx if proba[i] > 0]

    primary = top_preds[0]["disease"]
    info    = get_info(primary)

    try:
        db.session.add(Prediction(
            user_id    = session["user_id"],
            symptoms   = ", ".join(matched),
            predicted  = primary,
            confidence = top_preds[0]["confidence"]
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify({"predicted_disease": primary,
                    "confidence":        top_preds[0]["confidence"],
                    "top_predictions":   top_preds,
                    "matched_symptoms":  matched,
                    "unrecognized":      unrecognized,
                    **info})

# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT API
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are MedBot, an AI medical assistant inside the AI Virtual Doctor app.
Help users understand diseases, symptoms, medications, diet, and precautions.
Be clear, concise, and empathetic. Add disclaimer when needed.
Never diagnose — always recommend consulting a real doctor.
Respond in the same language the user uses (English or Tamil)."""

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    if not groq_client:
        return jsonify({"error": "Chatbot not configured. Set GROQ_API_KEY."}), 500
    data = request.get_json(force=True)
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_msg = data["message"].strip()
    history  = data.get("history", [])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, max_tokens=512, temperature=0.7
        )
        reply = response.choices[0].message.content.strip()

        try:
            db.session.add(ChatHistory(user_id=session["user_id"], role="user",      message=user_msg))
            db.session.add(ChatHistory(user_id=session["user_id"], role="assistant", message=reply))
            db.session.commit()
        except Exception:
            db.session.rollback()

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print(f"  AI Virtual Doctor — http://127.0.0.1:{port}")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=port)
