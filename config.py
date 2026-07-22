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
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "@@@jeeva18ani###")
    MYSQL_DB       = os.environ.get("MYSQL_DB",     "ai_doctor")
    MYSQL_CURSORCLASS = "DictCursor"

    # ── Groq LLM ───────────────────────────────────────────────────────────────
    # Free API key: https://console.groq.com
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
    GROQ_MODEL   = "llama-3.3-70b-versatile"    # latest stable model
