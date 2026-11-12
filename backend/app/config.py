from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"
MODEL_DIR = BACKEND_DIR / "models"
REPORT_DIR = BACKEND_DIR / "reports"

# Load optional local configuration. Exported environment variables still take precedence.
load_dotenv(PROJECT_DIR / ".env", override=False)
load_dotenv(BACKEND_DIR / ".env", override=False)

# DATABASE_PATH can be absolute or relative to the project root.
_db_value = os.getenv("DATABASE_PATH", "backend/data/supportiq.db")
_db_candidate = Path(_db_value)
DATABASE_PATH = _db_candidate if _db_candidate.is_absolute() else PROJECT_DIR / _db_candidate

MODEL_BUNDLE_PATH = MODEL_DIR / "supportiq_models.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
SENTIMENT_MODEL_ID = os.getenv(
    "SENTIMENT_MODEL_ID",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
)

_default_origins = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "http://localhost:8000,"
    "https://aryns293.github.io,"
    "https://supportiq-rndw.onrender.com"
)
_origins = os.getenv("CORS_ORIGINS", _default_origins)
CORS_ORIGINS = [origin.strip() for origin in _origins.split(",") if origin.strip()]
