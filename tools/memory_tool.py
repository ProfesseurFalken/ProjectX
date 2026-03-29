"""
ProjectX - Outil de Mémoire Explicite
Permet à Joshua de sauvegarder et retrouver des informations importantes
à la demande (nom de l'utilisateur, préférences, faits, etc.).

Stockage : fichier JSON dédié (data/explicit_memory.json) séparé de la
mémoire long-terme automatique (learning.py).

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-29
"""

import json
from pathlib import Path
from datetime import datetime

from langchain_core.tools import tool

from config import DATA_DIR

EXPLICIT_MEMORY_PATH = DATA_DIR / "explicit_memory.json"


def _load_memory() -> list[dict]:
    """Charge la mémoire explicite depuis le fichier JSON."""
    if EXPLICIT_MEMORY_PATH.exists():
        try:
            with open(EXPLICIT_MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_memory(entries: list[dict]) -> None:
    """Sauvegarde la mémoire explicite sur disque."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(EXPLICIT_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


@tool
def save_memory(content: str, category: str = "general") -> str:
    """Sauvegarde une information importante en mémoire permanente.

    Utilise cet outil pour mémoriser des faits importants que tu veux
    retenir entre les conversations : nom de l'utilisateur, préférences,
    informations apprises, etc.

    Args:
        content: L'information à mémoriser (ex: "L'utilisateur s'appelle Professeur Falken")
        category: Catégorie optionnelle (ex: "utilisateur", "fait", "preference", "general")
    """
    entries = _load_memory()
    entries.append({
        "content": content,
        "category": category,
        "saved_at": datetime.now().isoformat(),
    })
    _save_memory(entries)
    return f"Mémorisé : {content}"


@tool
def recall_memory(query: str = "") -> str:
    """Retrouve les informations sauvegardées en mémoire permanente.

    Utilise cet outil quand tu as besoin de retrouver des informations
    mémorisées auparavant : nom de l'utilisateur, préférences, faits, etc.

    Args:
        query: Mot-clé optionnel pour filtrer les souvenirs. Si vide, retourne tout.
    """
    entries = _load_memory()
    if not entries:
        return "Aucun souvenir en mémoire."

    if query:
        # Recherche par mots individuels (chaque mot du query est testé)
        query_words = query.lower().split()
        filtered = []
        for e in entries:
            text = (e["content"] + " " + e.get("category", "")).lower()
            if any(word in text for word in query_words):
                filtered.append(e)
        # Si aucun mot ne matche, retourner TOUS les souvenirs
        # pour que le LLM puisse trouver l'info lui-même
        if filtered:
            entries = filtered

    lines = []
    for e in entries:
        cat = e.get("category", "general")
        lines.append(f"[{cat}] {e['content']}")

    return "Souvenirs en mémoire :\n" + "\n".join(lines)
