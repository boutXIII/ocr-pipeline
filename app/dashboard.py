# =============================================================
# 🧠 docTR OCR Dashboard - Gestion des services locaux
# =============================================================
# Auteur : Valentin 🏴‍☠️ - Version 2025 Stable
# =============================================================

import os
import time
import datetime
import shutil
import psutil
import subprocess
from pathlib import Path
import streamlit as st
import json

# -------------------------------------------------------------
# 📂 Répertoires & chemins
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"

API_LOG = LOG_DIR / "api_log.txt"
UI_LOG = LOG_DIR / "ui_log.txt"
PID_FILE = LOG_DIR / "service_pids.json"

# -------------------------------------------------------------
# 🧩 Gestion des PID (suivi et nettoyage)
# -------------------------------------------------------------
def save_pids(pids: dict):
    """Sauvegarde les PID des services dans un fichier JSON"""
    import json
    with open(PID_FILE, "w", encoding="utf-8") as f:
        json.dump(pids, f, indent=2)

def load_pids() -> dict:
    """Charge les PID enregistrés"""
    import json
    if PID_FILE.exists():
        try:
            with open(PID_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def register_pid(service: str, pid: int):
    """Ajoute un PID au fichier de suivi"""
    pids = load_pids()
    pids[service] = pid
    save_pids(pids)

def stop_service_by_pid(service: str):
    """Arrête le service via son PID enregistré"""
    pids = load_pids()
    pid = pids.get(service)
    if not pid:
        return False
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(3)
        del pids[service]
        save_pids(pids)
        return True
    except Exception:
        return False

def clean_zombie_pids():
    """Nettoie les PID morts du fichier"""
    pids = load_pids()
    updated = {}
    for service, pid in pids.items():
        if psutil.pid_exists(pid):
            updated[service] = pid
    save_pids(updated)

# ==========================
# ⚙️ GESTION DES SERVICES
# ==========================
def rotate_log(log_path: Path, label: str):
    """Archive le log existant avec un timestamp."""
    if log_path.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = log_path.with_name(f"{log_path.stem}_{timestamp}{log_path.suffix}")
        try:
            shutil.move(log_path, archived)
            st.info(f"🧾 Log {label} archivé → {archived.name}")
        except Exception as e:
            st.warning(f"⚠️ Impossible d'archiver {label} : {e}")

def is_service_running(service_name: str) -> bool:
    """Vérifie si un service est actif via son PID"""
    pids = load_pids()
    pid = pids.get(service_name)
    return psutil.pid_exists(pid) if pid else False

def start_service(label: str, command: list, log_path: Path, service_name: str):
    """Démarre un service avec rotation du log"""
    # Stoppe service existant
    stop_service_by_pid(service_name)
    rotate_log(log_path, label)
    time.sleep(1)

    # Lance le processus
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            proc = subprocess.Popen(command, stdout=log_file, stderr=log_file, cwd=BASE_DIR)
            register_pid(service_name, proc.pid)
        st.success(f"✅ {label} lancé (PID {proc.pid})")
    except Exception as e:
        st.error(f"❌ Erreur au démarrage de {label} : {e}")


def tail_log(log_path: Path, lines: int = 20) -> str:
    """Lit les dernières lignes d’un fichier de log"""
    if not log_path.exists():
        return "(aucun log disponible)"
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.readlines()
        return "".join(content[-lines:]).strip()

# -------------------------------------------------------------
# 🧹 Nettoyage automatique des PIDs obsolètes
# -------------------------------------------------------------
clean_zombie_pids()

# -------------------------------------------------------------
# 🖥️ Interface Streamlit
# -------------------------------------------------------------
st.set_page_config(page_title="🧠 OCR Control Panel", layout="wide")
st.title("🧠 OCR Control Panel")
st.markdown("Gérez vos services **docTR** localement (API & Interface)")

col1, col2 = st.columns(2)

# --- Vérifie l’état des services
api_running = is_service_running("API FastAPI")
ui_running = is_service_running("Interface Streamlit")

# -------------------------------------------------------------
# ⚙️ API CONTROL
# -------------------------------------------------------------
with col1:
    st.subheader("⚙️ Service API (FastAPI)")
    if api_running:
        st.success("🟢 En cours d’exécution sur **http://10.8.197.100:8080/docs**")
        if st.button("⛔ Arrêter l’API"):
            if stop_service_by_pid("API FastAPI"):
                st.warning("🛑 API arrêtée avec succès.")
                st.rerun()
            else:
                st.error("⚠️ Impossible d’arrêter l’API.")
    else:
        st.warning("🔴 Arrêté")
        if st.button("🚀 Démarrer l’API"):
            start_service(
                "API FastAPI",
                [str(VENV_PYTHON),
                    "-m", "uvicorn",
                    "app.app_api:app",
                    "--host", "0.0.0.0",
                    "--port", "8080",
                    "--log-level", "trace",
                    "--reload",
                ],
                API_LOG,
                "API FastAPI"
            )
            st.rerun()

    with st.expander("📜 Logs API", expanded=False):
        st.text(tail_log(API_LOG, lines=30))

# -------------------------------------------------------------
# 🖼️ UI CONTROL
# -------------------------------------------------------------
with col2:
    st.subheader("🖼️ Interface Utilisateur (Streamlit)")
    if ui_running:
        st.success("🟢 En cours d’exécution sur **http://10.8.197.100:8502**")
        if st.button("⛔ Arrêter l’interface"):
            if stop_service_by_pid("Interface Streamlit"):
                st.warning("🛑 Interface arrêtée avec succès.")
                st.rerun()
            else:
                st.error("⚠️ Impossible d’arrêter l’interface.")
    else:
        st.warning("🔴 Arrêtée")
        if st.button("🚀 Démarrer l’interface"):
            start_service(
                "Interface Streamlit",
                [
                    str(VENV_PYTHON),
                    "-m", "streamlit",
                    "run", "app/app_ui.py",
                    "--server.address", "0.0.0.0",
                    "--server.port", "8502",
                    "--server.headless", "true"
                ],
                UI_LOG,
                "Interface Streamlit"
            )
            st.rerun()

    with st.expander("📜 Logs UI", expanded=False):
        st.text(tail_log(UI_LOG, lines=30))

st.divider()

# -----------------------------------------------
# 🔗 Liens directs
# -----------------------------------------------
st.divider()
st.markdown("""
### 🔗 Accès rapide :
- 📡 **API (FastAPI)** → [http://localhost:8080/docs](http://localhost:8080/docs)
- 🌐 **Interface Streamlit** → [http://localhost:8502](http://localhost:8502)
""")
