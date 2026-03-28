"""
ProjectX - Mémoire Persistante (Checkpointing + Store)
Gère la persistance des données de l'agent à deux niveaux :

1. MOYEN-TERME (SqliteSaver) : Sauvegarde automatique de l'état du graphe
   après chaque noeud. Permet de reprendre une conversation après un
   redémarrage. Chaque conversation est identifiée par un thread_id unique.

2. LONG-TERME (InMemoryStore + fichier JSON) : Stockage des apprentissages
   cross-session — préférences utilisateur, stratégies qui fonctionnent,
   erreurs à éviter, faits appris. Persiste sur disque via sérialisation JSON.

Architecture mémoire :
    ┌─ Court-terme ──────────────────────────────────┐
    │  state["messages"] : conversation en cours      │
    │  Automatique via LangGraph, pas géré ici        │
    └────────────────────────────────────────────────┘
                         ↓
    ┌─ Moyen-terme ──────────────────────────────────┐
    │  SqliteSaver : checkpoints par thread_id        │
    │  Fichier : data/checkpoints.sqlite              │
    └────────────────────────────────────────────────┘
                         ↓
    ┌─ Long-terme ───────────────────────────────────┐
    │  InMemoryStore : apprentissages cross-session   │
    │  Sérialisé : data/long_term_memory.json         │
    └────────────────────────────────────────────────┘

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import json
import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from config import CHECKPOINT_DB_PATH, LONG_TERM_MEMORY_PATH, DATA_DIR


# =============================================================================
# MÉMOIRE MOYEN-TERME : CHECKPOINTING DES CONVERSATIONS
# =============================================================================

async def get_checkpointer() -> AsyncSqliteSaver:
    """Crée et retourne le checkpointer SQLite asynchrone pour la persistance.

    Utilise AsyncSqliteSaver (basé sur aiosqlite) car Chainlit fonctionne
    en mode asynchrone. Le checkpointer sauvegarde automatiquement l'état
    du graphe LangGraph après chaque noeud exécuté. Cela permet :
    - De reprendre une conversation après un redémarrage de l'agent
    - De consulter l'historique complet des échanges
    - De faire du "time-travel" (revenir à un état précédent)

    Le fichier SQLite est créé automatiquement dans data/checkpoints.sqlite.

    Cette fonction est async car AsyncSqliteSaver.__init__ nécessite
    un event loop actif (asyncio.get_running_loop()).

    Returns:
        Une instance AsyncSqliteSaver connectée à la base SQLite locale.
    """
    # Création du répertoire data/ s'il n'existe pas
    DATA_DIR.mkdir(exist_ok=True)

    # AsyncSqliteSaver nécessite un event loop actif lors de l'instanciation.
    # On utilise le constructeur direct avec une connexion aiosqlite.
    # aiosqlite est déjà importé au niveau du module.
    checkpointer = AsyncSqliteSaver(conn=aiosqlite.connect(CHECKPOINT_DB_PATH))

    return checkpointer


# =============================================================================
# MÉMOIRE LONG-TERME : APPRENTISSAGES CROSS-SESSION
# =============================================================================

def get_memory_store() -> InMemoryStore:
    """Crée et retourne le store pour la mémoire long-terme de l'agent.

    Le store permet de sauvegarder des apprentissages qui persistent
    entre les sessions et les conversations :
    - Préférences de l'utilisateur (langue, style, sujets d'intérêt)
    - Stratégies qui ont fonctionné (séquences d'outils efficaces)
    - Erreurs à éviter (approches qui ont échoué et pourquoi)
    - Faits appris (informations découvertes lors des recherches)

    Les données sont chargées depuis un fichier JSON au démarrage
    et sauvegardées après chaque modification.

    Returns:
        Une instance InMemoryStore pré-chargée avec les apprentissages
        existants (ou vide si c'est la première utilisation).
    """
    store = InMemoryStore()

    # Chargement des données existantes depuis le fichier JSON
    # Si le fichier existe, on restaure les apprentissages précédents
    memory_path = Path(LONG_TERM_MEMORY_PATH)
    if memory_path.exists():
        try:
            with open(memory_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            # Restauration de chaque entrée dans le store
            # Le format JSON est : { "namespace|key": {"value": ..., "namespace": ..., "key": ...} }
            for entry_id, entry_data in saved_data.items():
                namespace = tuple(entry_data["namespace"])
                key = entry_data["key"]
                value = entry_data["value"]
                store.put(namespace, key, value)

        except (json.JSONDecodeError, KeyError) as e:
            # Si le fichier est corrompu, on repart de zéro
            # Mieux vaut perdre les apprentissages que planter l'agent
            print(f"[Mémoire] Avertissement : fichier mémoire corrompu, "
                  f"réinitialisation. Erreur : {e}")

    return store


def save_memory_store(store: InMemoryStore) -> None:
    """Sauvegarde le contenu du store en mémoire long-terme sur disque (JSON).

    Cette fonction est appelée après chaque réflexion de l'agent pour
    persister les nouveaux apprentissages. Le format JSON est lisible
    par un humain pour faciliter le debug.

    Args:
        store: L'instance InMemoryStore à sauvegarder.
    """
    # Extraction de toutes les entrées du store
    # On parcourt les namespaces connus pour sérialiser chaque item
    all_data = {}

    # Le InMemoryStore stocke les données dans un dict interne
    # On accède directement à la structure interne pour la sérialisation
    # Note : cette approche dépend de l'implémentation de InMemoryStore
    # et pourrait nécessiter un ajustement si l'API change
    try:
        # Parcours de toutes les entrées du store via search
        # On utilise un namespace vide pour tout récupérer
        # IMPORTANT : ne PAS passer query= car InMemoryStore sans index
        # d'embeddings lève une erreur sur les requêtes sémantiques
        items = store.search((), limit=1000)

        for item in items:
            entry_id = f"{'|'.join(item.namespace)}|{item.key}"
            all_data[entry_id] = {
                "namespace": list(item.namespace),
                "key": item.key,
                "value": item.value,
                "saved_at": datetime.now().isoformat(),
            }
    except Exception:
        # Si search() échoue (store vide ou API différente),
        # on sauvegarde un dict vide plutôt que de planter
        pass

    # Écriture du fichier JSON avec indentation pour lisibilité
    DATA_DIR.mkdir(exist_ok=True)
    with open(LONG_TERM_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)


async def get_conversation_history(checkpointer: AsyncSqliteSaver, thread_id: str) -> list:
    """Récupère l'historique des messages d'une conversation donnée.

    Utile pour le noeud "recall" qui a besoin de consulter les conversations
    passées afin d'y trouver du contexte pertinent.

    Cette fonction est async car AsyncSqliteSaver n'expose que des
    méthodes asynchrones (aget_tuple, aput, etc.).

    Args:
        checkpointer: Le AsyncSqliteSaver contenant les checkpoints.
        thread_id: L'identifiant unique de la conversation à récupérer.

    Returns:
        Liste des messages de la conversation, ou liste vide si la
        conversation n'existe pas.
    """
    try:
        config: dict = {"configurable": {"thread_id": thread_id}}

        # aget_tuple retourne le dernier checkpoint pour ce thread_id
        # (méthode async du AsyncSqliteSaver)
        checkpoint_tuple = await checkpointer.aget_tuple(config)  # type: ignore[arg-type]

        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            # Les messages sont stockés dans le channel "messages" de l'état
            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            return channel_values.get("messages", [])

    except Exception:
        pass

    return []
