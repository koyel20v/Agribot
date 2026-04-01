# config.py — Central configuration for AgriGenius backend
# All secrets/settings are loaded from environment variables (use a .env file locally)

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present

# ── MySQL ──────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 3306))
DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "koye20@%")          # ← no hardcoded password
DB_NAME     = os.getenv("DB_NAME", "agrigenius_db")

# ── JWT ────────────────────────────────────────────
JWT_SECRET       = os.getenv("JWT_SECRET", "change_this_in_production")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", 24))

# ── App ────────────────────────────────────────────
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
DEBUG    = os.getenv("DEBUG", "true").lower() == "true"