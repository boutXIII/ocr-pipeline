@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title 🧠 Installation silencieuse docTR OCR (CPU only)

set LOGFILE=%cd%\install_log.txt
if exist "%LOGFILE%" del "%LOGFILE%"

echo ===========================================
echo 📁 Installation silencieuse en cours...
echo 📄 Tous les détails seront enregistrés dans : %LOGFILE%
echo ===========================================

:: --- Vérifie Python ---
where python >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé ou pas dans le PATH. >> "%LOGFILE%"
    echo 👉 Télécharge-le depuis https://www.python.org/downloads/
    pause
    exit /b
)

:: --- Crée ou active le venv ---
if not exist venv (
    echo 🧱 Création de l'environnement virtuel... >> "%LOGFILE%"
    python -m venv venv >> "%LOGFILE%" 2>&1
)
call venv\Scripts\activate >nul 2>&1

:: --- Mise à jour pip / setuptools / wheel ---
echo 🔄 Mise à jour des outils Python... >> "%LOGFILE%"
python -m ensurepip --upgrade >> "%LOGFILE%" 2>&1
python -m pip install --upgrade pip setuptools wheel >> "%LOGFILE%" 2>&1

:: --- Installation des dépendances ---
echo ⚙️ Installation des dépendances (CPU only)... >> "%LOGFILE%"
pip install --no-cache-dir -r app\requirements.txt >> "%LOGFILE%" 2>&1

:: --- Vérifie si erreur ---
if errorlevel 1 (
    echo ❌ Erreur détectée pendant l'installation. Consulte "%LOGFILE%" pour les détails.
    pause
    exit /b
)

:: --- Résumé ---
echo ===========================================
echo ✅ Installation terminée avec succès
echo ===========================================
echo 🧠 docTR est prêt à l’emploi :
echo - Interface : streamlit run app/doctr_ui.py
echo - API REST  : uvicorn app.doctr_api:app --reload --port 8080
echo ===========================================
echo (voir "%LOGFILE%" pour les détails)
echo.

pause
