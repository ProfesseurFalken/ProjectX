"""
ProjectX - Outil d'Exécution de Code Python (Sandboxé)
Permet à l'agent d'écrire et d'exécuter du code Python dynamiquement.
Le code est exécuté dans un SUBPROCESS ISOLÉ avec capture de stdout/stderr
et un timeout de sécurité pour éviter les boucles infinies.

SÉCURITÉ : Le code est exécuté dans un processus Python SÉPARÉ, ce qui
garantit :
- Isolation mémoire : le code ne peut pas accéder aux variables de l'agent
- Isolation processus : un crash ou une boucle infinie ne bloque pas l'agent
- Timeout fiable : le subprocess est tué (kill) après le timeout
- Pas de pollution de l'état global (imports, variables, etc.)

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import subprocess
import sys
import tempfile
import os

from langchain_core.tools import tool

from config import CODE_EXECUTION_TIMEOUT


@tool
def execute_python(code: str) -> str:
    """Exécute du code Python dans un subprocess isolé et retourne le résultat.

    Utilise cette fonction UNIQUEMENT pour : calculs, manipulation de données,
    test d'algorithmes, génération de fichiers.
    N'utilise PAS cette fonction pour accéder à internet — utilise web_search
    ou scrape_webpage à la place.

    Le code est exécuté dans un processus Python séparé (sandbox) avec un
    timeout strict. La sortie standard (print) est capturée et retournée.

    Args:
        code: Le code Python à exécuter. Peut être sur plusieurs lignes.
              Exemples :
              - "print(2 + 2)"
              - "import math\\nprint(math.pi)"
              - Un script complet avec fonctions, classes, etc.

    Returns:
        La sortie du code (tout ce qui est affiché par print()), les
        éventuelles erreurs, et un indicateur de succès ou d'échec.
        En cas de timeout, le processus est tué et un message d'erreur
        est retourné.
    """
    # --- Vérification de sécurité : bloquer l'accès web via execute_python ---
    # L'agent doit utiliser web_search / scrape_webpage pour accéder au web,
    # pas execute_python avec requests/urllib/httpx. On bloque ces imports
    # pour forcer le bon comportement.
    import re as _re
    web_modules = ["requests", "urllib", "httpx", "aiohttp", "http.client"]
    for mod in web_modules:
        # Détecte import mod, from mod, __import__('mod')
        if _re.search(rf'(?:^|\s|;)(?:import\s+{mod}|from\s+{mod})', code):
            return (
                f"Erreur : l'utilisation de '{mod}' est bloquée dans execute_python. "
                f"Pour accéder à internet, utilise l'outil **web_search** (recherche) "
                f"ou **scrape_webpage** (lecture de page web) à la place."
            )

    # --- Écriture du code dans un fichier temporaire ---
    # On utilise un fichier temporaire plutôt que -c pour supporter le code
    # multilignes complexe sans problème d'échappement
    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="joshua_exec_",
            delete=False,
            encoding="utf-8",
        )
        tmp_file.write(code)
        tmp_file.close()

        # --- Exécution dans un subprocess isolé ---
        # Le code tourne dans un processus Python complètement séparé :
        # - Pas d'accès aux variables/objets de l'agent
        # - Un crash ne tue pas l'agent
        # - Le timeout kill le processus proprement
        result = subprocess.run(
            [sys.executable, tmp_file.name],
            capture_output=True,       # Capture stdout + stderr
            text=True,                 # Sortie en str (pas bytes)
            timeout=CODE_EXECUTION_TIMEOUT,
            cwd=tempfile.gettempdir(), # Répertoire de travail isolé
            env={                      # Environnement minimal
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",      # Pas d'accès aux modules du projet
                "PYTHONIOENCODING": "utf-8",
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),  # Nécessaire sur Windows
                "TEMP": tempfile.gettempdir(),
                "TMP": tempfile.gettempdir(),
            },
        )

        # --- Construction de la réponse ---
        response_parts = [f"Code exécuté :\n```python\n{code}\n```"]

        # Sortie standard (résultat des print())
        stdout = result.stdout.strip()
        if stdout:
            # Troncature si la sortie est trop longue
            if len(stdout) > 5000:
                stdout = stdout[:5000] + "\n\n[... sortie tronquée]"
            response_parts.append(f"Sortie :\n{stdout}")

        # Sortie d'erreur (exceptions, warnings)
        stderr = result.stderr.strip()
        if stderr:
            if len(stderr) > 3000:
                stderr = stderr[:3000] + "\n\n[... erreur tronquée]"
            response_parts.append(f"Erreur :\n{stderr}")

        # Code de retour
        if result.returncode != 0 and not stderr:
            response_parts.append(
                f"Le processus s'est terminé avec le code {result.returncode}."
            )
        elif not stdout and not stderr and result.returncode == 0:
            response_parts.append(
                "Le code s'est exécuté sans erreur et sans produire de sortie."
            )

        return "\n\n".join(response_parts)

    except subprocess.TimeoutExpired:
        return (
            f"Code exécuté :\n```python\n{code}\n```\n\n"
            f"Erreur : Timeout — le code a dépassé la limite de "
            f"{CODE_EXECUTION_TIMEOUT} secondes et le processus a été tué. "
            f"Vérifiez qu'il n'y a pas de boucle infinie."
        )
    except Exception as e:
        return (
            f"Code exécuté :\n```python\n{code}\n```\n\n"
            f"Erreur lors de l'exécution : {str(e)}"
        )
    finally:
        # Nettoyage du fichier temporaire
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
