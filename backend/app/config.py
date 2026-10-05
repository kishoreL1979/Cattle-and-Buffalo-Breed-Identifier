import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Load .env file from backend directory, workspace root, or current directory
for env_file in [BASE_DIR / "backend" / ".env", BASE_DIR / ".env", Path.cwd() / ".env"]:
    if env_file.exists():
        load_dotenv(env_file, override=True)

MODEL_DIR = BASE_DIR / "models" / "cnn"
MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"

# OpenRouter Configuration (Supports both OPENROUTER_API_KEY and openrouter.api.key)
OPENROUTER_API_KEY = (os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter.api.key") or "").strip()
OPENROUTER_MODEL = (os.getenv("OPENROUTER_MODEL") or os.getenv("openrouter.model") or "google/gemini-2.5-flash").strip()

base_url = (os.getenv("OPENROUTER_URL") or os.getenv("openrouter.base-url") or "https://openrouter.ai/api/v1").strip().rstrip("/")
if not base_url.endswith("/chat/completions"):
    OPENROUTER_URL = f"{base_url}/chat/completions"
else:
    OPENROUTER_URL = base_url

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
