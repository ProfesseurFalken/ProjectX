"""
ProjectX - Outil de Statut des Tâches
Permet à l'utilisateur et à l'agent de suivre les tâches en cours,
savoir si l'agent travaille en arrière-plan, et voir l'historique récent.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-30
"""

import time
import threading
from collections import deque

from langchain_core.tools import tool

# Historique des tâches (thread-safe via deque)
_MAX_HISTORY = 50
_task_history: deque[dict] = deque(maxlen=_MAX_HISTORY)
_current_task: dict | None = None
_lock = threading.Lock()


def task_start(name: str, detail: str = "") -> None:
    """Enregistre le début d'une tâche (appelé par le code interne, pas par le LLM)."""
    global _current_task
    with _lock:
        _current_task = {
            "name": name,
            "detail": detail,
            "start_time": time.time(),
            "status": "en cours",
        }


def task_end(name: str, result: str = "terminé") -> None:
    """Enregistre la fin d'une tâche."""
    global _current_task
    with _lock:
        if _current_task and _current_task["name"] == name:
            _current_task["status"] = result
            _current_task["end_time"] = time.time()
            _current_task["duration"] = round(
                _current_task["end_time"] - _current_task["start_time"], 1
            )
            _task_history.append(dict(_current_task))
            _current_task = None
        else:
            _task_history.append({
                "name": name,
                "status": result,
                "end_time": time.time(),
            })


@tool
def get_task_status() -> str:
    """Affiche la tâche en cours et l'historique récent des tâches de l'agent.

    Utilise cette fonction quand l'utilisateur demande ce que l'agent
    est en train de faire, s'il travaille en arrière-plan, ou pour
    voir les dernières actions effectuées.

    Returns:
        Statut de la tâche en cours + historique des 10 dernières tâches.
    """
    with _lock:
        lines = []

        # Tâche en cours
        if _current_task:
            elapsed = round(time.time() - _current_task["start_time"], 1)
            lines.append(
                f"⚙️ TÂCHE EN COURS : {_current_task['name']}"
                f" (depuis {elapsed}s)"
            )
            if _current_task.get("detail"):
                lines.append(f"   Détail : {_current_task['detail']}")
        else:
            lines.append("✅ Aucune tâche en cours — l'agent est disponible.")

        # Historique récent
        recent = list(_task_history)[-10:]
        if recent:
            lines.append(f"\n📋 Historique des {len(recent)} dernière(s) tâche(s) :")
            for t in reversed(recent):
                duration = t.get("duration", "?")
                lines.append(
                    f"  • {t['name']} — {t['status']} ({duration}s)"
                )
        else:
            lines.append("\n📋 Aucune tâche précédente dans l'historique.")

        return "\n".join(lines)
