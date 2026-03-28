"""
ProjectX - Persistance des Sessions
Permet à Joshua de sauvegarder et reprendre des conversations.
Chaque conversation est identifiée par un thread_id et stockée dans
un fichier JSON avec un titre et un horodatage.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import json
from datetime import datetime
from pathlib import Path

from config import SESSIONS_FILE


def _load_sessions() -> list[dict]:
    """Charge la liste des sessions depuis le fichier JSON."""
    path = Path(SESSIONS_FILE)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_sessions(sessions: list[dict]) -> None:
    """Sauvegarde la liste des sessions dans le fichier JSON."""
    path = Path(SESSIONS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def save_session(thread_id: str, first_message: str) -> None:
    """Enregistre ou met a jour une session.

    Args:
        thread_id: Identifiant unique de la conversation.
        first_message: Premier message de l'utilisateur (utilisé comme titre).
    """
    sessions = _load_sessions()

    # Vérifier si la session existe déjà
    for session in sessions:
        if session.get("thread_id") == thread_id:
            session["last_active"] = datetime.now().isoformat()
            _save_sessions(sessions)
            return

    # Nouvelle session
    title = first_message[:80].strip()
    if len(first_message) > 80:
        title += "..."

    sessions.append({
        "thread_id": thread_id,
        "title": title,
        "created": datetime.now().isoformat(),
        "last_active": datetime.now().isoformat(),
    })

    # Garder les 20 sessions les plus récentes
    sessions.sort(key=lambda s: s.get("last_active", ""), reverse=True)
    sessions = sessions[:20]

    _save_sessions(sessions)


def get_recent_sessions(limit: int = 5) -> list[dict]:
    """Retourne les sessions les plus récentes.

    Args:
        limit: Nombre maximum de sessions à retourner.

    Returns:
        Liste des sessions triées par date d'activité décroissante.
    """
    sessions = _load_sessions()
    sessions.sort(key=lambda s: s.get("last_active", ""), reverse=True)
    return sessions[:limit]


def delete_session(thread_id: str) -> bool:
    """Supprime une session par son thread_id.

    Args:
        thread_id: Identifiant de la session à supprimer.

    Returns:
        True si la session a été trouvée et supprimée, False sinon.
    """
    sessions = _load_sessions()
    original_len = len(sessions)
    sessions = [s for s in sessions if s.get("thread_id") != thread_id]
    if len(sessions) < original_len:
        _save_sessions(sessions)
        return True
    return False
