"""
AI Virtual Doctor — Flask App
Features: Disease Prediction + LLM Chatbot + MySQL Auth
"""
import os, pickle, ast
import numpy as np
import pandas as pd
import bcrypt
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash)
from flask_mysqldb import MySQL
from groq import Groq
from config import Config

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Fallback manual .env loader
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
app.config.from_object(Config)

# ── MySQL ──────────────────────────────────────────────────────────────────────
mysql = MySQL(app)

# ── Groq LLM ───────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=Config.GROQ_API_KEY)

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
    SYM_NORM_MAP[s.strip().lower()]              = s
    SYM_NORM_MAP[s.strip().lower().replace("_"," ")] = s

print(f"[OK] Model loaded | Symptoms: {len(SYMPTOMS)} | Classes: {len(le.classes_)}")

# ── Load CSV info maps ─────────────────────────────────────────────────────────
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
    prec_map[key] = [str(r.get(f"Precaution_{i}","")).strip()
                     for i in range(1,5)
                     if str(r.get(f"Precaution_{i}","")).strip() not in ("","nan")]

def _parse(raw):
    if isinstance(raw, list): return raw
    try:
        v = ast.literal_eval(str(raw))
        if isinstance(v, list): return [str(x).strip() for x in v]
    except: pass
    return [x.strip() for x in str(raw).split(",") if x.strip() and x.strip()!="nan"]

def get_info(disease):
    k = _norm(disease)
    return {
        "description": desc_map.get(k, "Description not available."),
        "diet":        _parse(diet_map.get(k, [])),
        "medications": _parse(med_map.get(k, [])),
        "precautions": prec_map.get(k, []),
        "workouts":    _parse(work_map.get(k, [])),
    }

# ── Auth helpers ───────────────────────────────────────────────────────────────
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

@app.route("/login", methods=["GET","POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email    = request.form.get("email","").strip()
        password = request.form.get("password","").strip()
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()
        if user and bcrypt.checkpw(password.encode(), user["password"].encode()):
            session["user_id"]   = user["id"]
            session["user_name"] = user["full_name"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name     = request.form.get("full_name","").strip()
        email    = request.form.get("email","").strip()
        password = request.form.get("password","").strip()
        if not all([name, email, password]):
            flash("All fields are required.", "error")
            return render_template("register.html")
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        try:
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO users (full_name, email, password) VALUES (%s,%s,%s)",
                        (name, email, hashed))
            mysql.connection.commit()
            cur.close()
            flash("Account created! Please login.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash("Email already registered.", "error")
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
    display_syms = [s.replace("_"," ") for s in SYMPTOMS]
    return render_template("index.html",
                           symptoms=display_syms,
                           user_name=session.get("user_name",""))

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION API
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/symptoms")
@login_required
def api_symptoms():
    return jsonify({"symptoms": [s.replace("_"," ") for s in SYMPTOMS]})

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
        col = SYM_NORM_MAP.get(sym) or SYM_NORM_MAP.get(sym.replace(" ","_"))
        if col:
            input_vec[SYMPTOMS.index(col)] = 1.0
            matched.append(col)
        else:
            unrecognized.append(sym)

    if input_vec.sum() == 0:
        return jsonify({"error": f"Symptoms not recognized: {unrecognized}"}), 400

    proba   = model.predict_proba([input_vec])[0]
    top_idx = np.argsort(proba)[::-1][:3]
    top_preds = [{"disease": le.inverse_transform([i])[0],
                  "confidence": round(float(proba[i])*100, 2)}
                 for i in top_idx if proba[i] > 0]

    primary = top_preds[0]["disease"]
    info    = get_info(primary)

    # Save to DB
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO predictions (user_id,symptoms,predicted,confidence) VALUES (%s,%s,%s,%s)",
            (session["user_id"], ", ".join(matched), primary, top_preds[0]["confidence"])
        )
        mysql.connection.commit()
        cur.close()
    except: pass

    return jsonify({"predicted_disease": primary,
                    "confidence": top_preds[0]["confidence"],
                    "top_predictions": top_preds,
                    "matched_symptoms": matched,
                    "unrecognized": unrecognized,
                    **info})

# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT API
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are MedBot, an AI medical assistant inside the AI Virtual Doctor app.
You help users understand diseases, symptoms, medications, diet, and precautions.
Always be clear, concise, and empathetic. Add a disclaimer when needed.
Never diagnose — always recommend consulting a real doctor for actual medical decisions.
Respond in the same language the user uses (English or Tamil)."""

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(force=True)
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_msg = data["message"].strip()
    history  = data.get("history", [])   # [{role, content}, ...]

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-10:]:              # last 10 turns for context
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        response = groq_client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()

        # Save to DB
        try:
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO chat_history (user_id,role,message) VALUES (%s,%s,%s)",
                        (session["user_id"], "user", user_msg))
            cur.execute("INSERT INTO chat_history (user_id,role,message) VALUES (%s,%s,%s)",
                        (session["user_id"], "assistant", reply))
            mysql.connection.commit()
            cur.close()
        except: pass

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("="*50)
    print(f"  AI Virtual Doctor — http://127.0.0.1:{port}")
    print("="*50)
    app.run(debug=False, host="0.0.0.0", port=port)
