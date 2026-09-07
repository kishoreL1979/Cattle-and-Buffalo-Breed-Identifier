import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Load .env file from backend directory or workspace root
env_path = BASE_DIR / "backend" / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv(BASE_DIR / ".env")

MODEL_DIR = BASE_DIR / "models" / "cnn"
MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Canonical breed mapping to Animal Type
BREED_TO_ANIMAL_TYPE = {
    "gir": "Cattle",
    "jaffrabadi": "Buffalo",
    "kankrej": "Cattle",
    "mehsana": "Buffalo",
    "murrah": "Buffalo",
    "red_sindhi": "Cattle",
    "sahiwal": "Cattle",
    "surti": "Buffalo",
    "tharparkar": "Cattle"
}

BREED_DISPLAY_NAMES = {
    "gir": "Gir",
    "jaffrabadi": "Jaffrabadi",
    "kankrej": "Kankrej",
    "mehsana": "Mehsana",
    "murrah": "Murrah",
    "red_sindhi": "Red Sindhi",
    "sahiwal": "Sahiwal",
    "surti": "Surti",
    "tharparkar": "Tharparkar"
}

SUPPORTED_BREED_NAMES = [
    "Gir", "Jaffrabadi", "Kankrej", "Mehsana",
    "Murrah", "Red Sindhi", "Sahiwal", "Surti", "Tharparkar"
]

MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
