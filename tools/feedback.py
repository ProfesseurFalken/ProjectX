"""
ProjectX - Système de Feedback Utilisateur
Collecte et stocke les évaluations de l'utilisateur (👍/👎) sur les réponses
de Joshua. Ces feedbacks alimentent le module de réflexion pour améliorer
les performances de l'agent au fil du temps.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-30
"""

import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

from config import DATA_DIR

logger = logging.getLogger(__name__)

_FEEDBACK_FILE = Path(DATA_DIR) / "feedback.json"
_lock = threading.Lock()

# Cache mémoire des derniers feedbacks (pour la réflexion)
_recent_feedbacks: deque = deque(maxlen=100)


def _load_feedbacks() -> list[dict]:
    """Charge les feedbacks depuis le fichier JSON."""
    if _FEEDBACK_FILE.exists():
        try:
            return json.loads(_FEEDBACK_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_feedbacks(feedbacks: list[dict]) -> None:
    """Sauvegarde les feedbacks sur disque."""
    _FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FEEDBACK_FILE.write_text(
        json.dumps(feedbacks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def store_feedback(
    rating: str,
    user_query: str,
    agent_response: str,
    specialist: str = "",
) -> None:
    """Enregistre un feedback utilisateur.

    Args:
        rating: "positive" ou "negative"
        user_query: La question de l'utilisateur
        agent_response: La réponse de l'agent (tronquée)
        specialist: Le spécialiste qui a répondu
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "rating": rating,
        "specialist": specialist,
        "query": user_query[:200],
        "response": agent_response[:500],
    }

    with _lock:
        feedbacks = _load_feedbacks()
        feedbacks.append(entry)
        # Garder les 500 derniers feedbacks max
        if len(feedbacks) > 500:
            feedbacks = feedbacks[-500:]
        _save_feedbacks(feedbacks)
        _recent_feedbacks.append(entry)

    logger.info(f"Feedback: {rating} pour {specialist} (query: {user_query[:50]}...)")


def get_recent_feedback_summary(limit: int = 20) -> str:
    """Retourne un résumé des derniers feedbacks pour la réflexion.

    Args:
        limit: Nombre de feedbacks récents à inclure.

    Returns:
        Texte formaté pour injection dans le prompt de réflexion.
    """
    with _lock:
        feedbacks = list(_recent_feedbacks)[-limit:]

    if not feedbacks:
        # Charger depuis le fichier si le cache mémoire est vide
        feedbacks = _load_feedbacks()[-limit:]

    if not feedbacks:
        return ""

    positive = sum(1 for f in feedbacks if f["rating"] == "positive")
    negative = sum(1 for f in feedbacks if f["rating"] == "negative")

    lines = [
        f"\n[FEEDBACK UTILISATEUR] {positive} 👍 / {negative} 👎 sur les {len(feedbacks)} dernières évaluations."
    ]

    # Détailler les feedbacks négatifs (plus utiles pour l'apprentissage)
    neg_feedbacks = [f for f in feedbacks if f["rating"] == "negative"]
    if neg_feedbacks:
        lines.append("Réponses mal notées :")
        for f in neg_feedbacks[-5:]:
            lines.append(
                f"  - [{f.get('specialist', '?')}] Q: {f['query'][:80]} → Réponse jugée insuffisante"
            )

    return "\n".join(lines)
