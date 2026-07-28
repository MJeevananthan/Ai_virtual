"""
Configuration — DB + Groq API
"""
import os

class Config:
    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("SECRET_KEY", "aivdoctor_super_secret_2026")

    # ── MySQL ──────────────────────────────────────────────────────────────────
    MYSQL_HOST     = os.environ.get("MYSQL_HOST",   "localhost")
    MYSQL_USER     = os.environ.get("MYSQL_USER",   "root")
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "your_mysql_password_here")
    MYSQL_DB       = os.environ.get("MYSQL_DB",     "ai_doctor")
    MYSQL_CURSORCLASS = "DictCursor"

    # ── Groq LLM ───────────────────────────────────────────────────────────────
    # Set GROQ_API_KEY environment variable or paste key in .env file
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
    GROQ_MODEL   = "llama-3.3-70b-versatile"

