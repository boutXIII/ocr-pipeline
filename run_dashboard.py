import os
import subprocess
import sys
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"
APP_PATH = BASE_DIR / "app" / "dashboard.py"

# Vérifie que le fichier existe
if not APP_PATH.exists():
    print(f"❌ Fichier introuvable : {APP_PATH}")
    sys.exit(1)

print("🚀 Démarrage du OCR Control Panel (Streamlit)...")

# Ouvre automatiquement le navigateur après un petit délai
webbrowser.open("http://localhost:8501", new=1)

# Lance Streamlit avec le dashboard
subprocess.call([str(VENV_PYTHON), "-m", "streamlit", "run", str(APP_PATH), "--server.port", "8501", "--server.headless", "false"])
