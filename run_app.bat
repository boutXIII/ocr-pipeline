@echo off
chcp 65001 >nul
title 🧠 OCR Control Panel

setlocal enabledelayedexpansion
set BASE_DIR=%~dp0
cd /d "%BASE_DIR%"

echo ============================================
echo 🧠 Lancement du tableau de bord docTR OCR
echo ============================================
echo.

if not exist venv (
    echo ❌ Environnement virtuel introuvable. Lancez install_env.bat d'abord.
    pause
    exit /b
)

call venv\Scripts\activate

echo 🚀 Ouverture du tableau de bord sur http://localhost:8501
python -m streamlit run app\dashboard.py --server.port 8501 --server.headless true
pause
