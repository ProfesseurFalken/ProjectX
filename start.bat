@echo off
REM =============================================================================
REM ProjectX - Script de lancement automatique
REM Lance le serveur Ollama (si nécessaire) puis l'agent Chainlit.
REM
REM Auteur  : ProfesseurFalken
REM Contact : wojcikej@orange.fr
REM GitHub  : https://github.com/ProfesseurFalken/ProjectX
REM =============================================================================

echo ============================================
echo   ProjectX - Agent AI Autonome Local
echo   Par ProfesseurFalken
echo ============================================
echo.

REM --- Vérification d'Ollama ---
echo [1/3] Verification d'Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR : Ollama n'est pas installe ou pas dans le PATH.
    echo Telecharger Ollama sur https://ollama.com
    pause
    exit /b 1
)
echo       Ollama OK.

REM --- Démarrage d'Ollama en arrière-plan (si pas déjà lancé) ---
echo [2/3] Demarrage du serveur Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo       Lancement d'Ollama en arriere-plan...
    start /min "" ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo       Ollama est deja en cours d'execution.
)

REM --- Activation du venv et lancement de Chainlit ---
echo [3/3] Lancement de ProjectX...
echo.
echo   Interface web : http://localhost:8000
echo   Appuie sur Ctrl+C pour arreter.
echo ============================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
chainlit run main.py
