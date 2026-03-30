"""
ProjectX - Système d'Auto-Apprentissage (Reflection & Recall)
Ce module implémente les deux noeuds clés du système d'apprentissage :

1. RECALL (Rappel) : Au début de chaque interaction, l'agent consulte sa
   mémoire long-terme pour retrouver les apprentissages pertinents à la
   requête de l'utilisateur. Ces apprentissages sont injectés dans le
   prompt système pour enrichir le contexte.

2. REFLECTION (Réflexion) : Périodiquement (toutes les N interactions),
   l'agent analyse ses dernières interactions et en extrait des leçons :
   - Préférences de l'utilisateur détectées
   - Stratégies d'outils qui ont fonctionné
   - Erreurs commises et comment les éviter
   - Faits importants appris

La réflexion est déclenchée automatiquement via un compteur d'interactions
stocké dans l'état du graphe.

Architecture du flux d'apprentissage :
    ┌────────────────────────────────────────────────────┐
    │  Nouvelle requête utilisateur                       │
    │       ↓                                             │
    │  [RECALL] Recherche mémoire long-terme pertinente   │
    │       ↓                                             │
    │  [AGENT] LLM avec contexte enrichi + outils         │
    │       ↓                                             │
    │  [TOOLS] Exécution des outils sélectionnés          │
    │       ↓                                             │
    │  [SHOULD_REFLECT?] Compteur >= REFLECTION_FREQUENCY? │
    │       ↓ OUI                    ↓ NON                │
    │  [REFLECTION] Analyse +     Retour à [AGENT]        │
    │  stockage apprentissages                            │
    │       ↓                                             │
    │  Retour à [AGENT]                                   │
    └────────────────────────────────────────────────────┘

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

from datetime import datetime
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    REFLECTION_FREQUENCY,
    MAX_RECALL_ITEMS,
)
from agent.memory import save_memory_store


# =============================================================================
# PROMPTS POUR LA RÉFLEXION
# =============================================================================

# Prompt utilisé par le noeud de réflexion pour analyser les interactions
# Le LLM va recevoir les derniers messages et devra en extraire des leçons
REFLECTION_PROMPT = """Tu es un système d'analyse qui extrait des apprentissages à partir de conversations.

Analyse les messages suivants entre un utilisateur et un assistant AI et extrais :

1. **PRÉFÉRENCES UTILISATEUR** : langue préférée, style de communication, sujets d'intérêt, formats de réponse préférés, etc.
2. **STRATÉGIES EFFICACES** : quelles séquences d'outils ont bien fonctionné ? Quelles approches ont résolu le problème efficacement ?
3. **ERREURS À ÉVITER** : quelles approches ont échoué ? Pourquoi ? Comment les éviter à l'avenir ?
4. **FAITS APPRIS** : informations factuelles importantes découvertes (résultats de recherche, données, configurations, etc.)

Réponds en JSON avec exactement cette structure :
{{
    "user_preferences": ["préférence 1", "préférence 2"],
    "successful_strategies": ["stratégie 1", "stratégie 2"],
    "failed_approaches": ["erreur 1 et sa raison", "erreur 2 et sa raison"],
    "learned_facts": ["fait 1", "fait 2"]
}}

Si une catégorie est vide, utilise une liste vide [].
Sois concis mais précis. Ne répète pas ce qui est déjà connu.

Messages à analyser :
{messages}
"""


# =============================================================================
# NOEUD RECALL : RAPPEL DE LA MÉMOIRE LONG-TERME
# =============================================================================

def recall_memories(state, store) -> dict:
    """Noeud RECALL : recherche et injecte les apprentissages pertinents.

    Ce noeud est exécuté au DÉBUT de chaque interaction dans le graphe.
    Il consulte la mémoire long-terme (InMemoryStore) pour trouver les
    apprentissages les plus pertinents par rapport à la requête actuelle
    de l'utilisateur, puis les injecte dans le contexte.

    Le rappel fonctionne par recherche dans le store avec le texte de
    la dernière requête comme clé de recherche.

    Args:
        state: L'état actuel du graphe LangGraph. Contient :
               - "messages" : liste des messages de la conversation
               - "recalled_memories" : chaîne à injecter dans le prompt
               - "interaction_count" : compteur d'interactions
        store: L'InMemoryStore contenant les apprentissages long-terme.

    Returns:
        L'état mis à jour avec "recalled_memories" rempli des
        apprentissages pertinents trouvés dans le store.
    """
    # Récupération du dernier message de l'utilisateur
    # C'est ce message qui sert de requête pour la recherche en mémoire
    messages = state.get("messages", [])
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # Si pas de message utilisateur, rien à rappeler
    if not last_user_msg:
        return state

    # --- Recherche dans la mémoire long-terme ---
    # On cherche dans toutes les catégories de mémoire
    recalled_items = []

    try:
        # Recherche dans les différentes catégories de mémoire
        # On cherche dans le namespace "learnings" sans paramètre query
        # car InMemoryStore sans index d'embeddings ne supporte pas
        # la recherche sémantique et lèverait une ValueError
        results = store.search(
            ("learnings",),
            limit=MAX_RECALL_ITEMS,
        )

        # Formatage des résultats trouvés pour injection dans le prompt
        for item in results:
            value = item.value
            if isinstance(value, dict):
                # Extraction des informations pertinentes de chaque catégorie
                for category in ["user_preferences", "successful_strategies",
                                 "failed_approaches", "learned_facts"]:
                    entries = value.get(category, [])
                    if entries:
                        category_label = {
                            "user_preferences": "Préférences connues",
                            "successful_strategies": "Stratégies efficaces",
                            "failed_approaches": "Erreurs à éviter",
                            "learned_facts": "Faits connus",
                        }.get(category, category)

                        for entry in entries:
                            recalled_items.append(f"[{category_label}] {entry}")

    except Exception:
        # Si la recherche échoue (store vide, erreur), on continue sans rappel
        # L'agent fonctionne très bien même sans mémoire long-terme
        pass

    # Construction du texte de rappel à injecter dans le prompt système
    if recalled_items:
        # On déduplique et on limite le nombre d'items
        unique_items = list(dict.fromkeys(recalled_items))[:MAX_RECALL_ITEMS * 2]
        recalled_text = "\n\nApprentissages de tes interactions passées :\n" + \
                        "\n".join(f"- {item}" for item in unique_items)
    else:
        recalled_text = ""

    # Mise à jour de l'état avec les souvenirs rappelés
    return {**state, "recalled_memories": recalled_text}


# =============================================================================
# NOEUD REFLECTION : ANALYSE ET APPRENTISSAGE
# =============================================================================

async def reflect_on_interactions(state, store) -> dict:
    """Noeud REFLECTION : analyse les interactions récentes et extrait des leçons.

    Ce noeud est déclenché périodiquement (toutes les REFLECTION_FREQUENCY
    interactions). Il utilise le LLM pour analyser les derniers échanges
    et en extraire :
    - Les préférences de l'utilisateur
    - Les stratégies qui ont fonctionné
    - Les erreurs à éviter
    - Les faits importants appris

    Les apprentissages sont stockés dans le InMemoryStore et sauvegardés
    sur disque (fichier JSON) pour persister entre les redémarrages.

    Args:
        state: L'état actuel du graphe contenant les messages récents.
        store: L'InMemoryStore où stocker les apprentissages.

    Returns:
        L'état avec le compteur d'interactions remis à zéro.
    """
    messages = state.get("messages", [])

    # On prend les derniers messages pour la réflexion
    # Pas tous les messages pour ne pas saturer le contexte Ollama
    recent_messages = messages[-20:]  # 20 derniers messages max

    # --- Compression des messages en texte pour le prompt ---
    # On convertit les objets Message en texte lisible
    messages_text = ""
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            messages_text += f"UTILISATEUR : {msg.content}\n"
        elif isinstance(msg, ToolMessage):
            # --- Résultats des outils (scraping, recherche, etc.) ---
            # C'est ici que se trouvent les données brutes lues par Joshua.
            # On les inclut dans la réflexion pour que l'apprentissage
            # capture les faits découverts via le web, pas seulement
            # ce que l'agent a résumé dans sa réponse.
            tool_name = getattr(msg, "name", "outil")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 3000:
                content = content[:3000] + "..."
            messages_text += f"RÉSULTAT OUTIL [{tool_name}] : {content}\n"
        elif isinstance(msg, AIMessage):
            # Pour les messages de l'assistant, on résume le contenu
            # (les tool_calls peuvent être très verbeux)
            content = msg.content if msg.content else "[appel d'outil]"
            # S'assurer que content est une chaîne (peut être une liste dans certains cas)
            if not isinstance(content, str):
                content = str(content)
            # Troncature des réponses trop longues (1500 chars pour garder
            # assez de contexte tout en restant dans le budget du LLM)
            if len(content) > 1500:
                content = content[:1500] + "..."
            messages_text += f"ASSISTANT : {content}\n"

    # Si pas assez de messages pour une réflexion utile, on skip
    if len(messages_text.strip()) < 50:
        return {**state, "interaction_count": 0}

    # --- Injection du feedback utilisateur dans la réflexion ---
    from tools.feedback import get_recent_feedback_summary
    feedback_summary = get_recent_feedback_summary(limit=20)
    if feedback_summary:
        messages_text += f"\n{feedback_summary}\n"

    try:
        # --- Appel au LLM pour l'analyse ---
        # On utilise une instance séparée du LLM avec une température basse
        # pour des analyses plus déterministes et structurées
        reflection_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,  # Très déterministe pour l'analyse
        )

        # Construction du prompt avec les messages à analyser
        prompt = REFLECTION_PROMPT.format(messages=messages_text)

        # Invocation async du LLM pour la réflexion
        response = await reflection_llm.ainvoke([HumanMessage(content=prompt)])
        response_text = response.content
        # S'assurer que response_text est une chaîne
        if not isinstance(response_text, str):
            response_text = str(response_text)

        # --- Parsing de la réponse JSON ---
        # Le LLM devrait retourner du JSON, mais on gère les cas d'erreur
        import json

        # Extraction du JSON du texte de la réponse
        # Le LLM peut entourer le JSON de texte ou de blocs markdown
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            learnings = json.loads(json_str)
        else:
            # Pas de JSON trouvé — on skip cette réflexion
            return {**state, "interaction_count": 0}

        # --- Stockage des apprentissages dans le store ---
        # Chaque réflexion est stockée avec un timestamp comme clé
        # dans le namespace "learnings"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Vérification que les apprentissages ne sont pas tous vides
        has_content = any(
            learnings.get(cat, [])
            for cat in ["user_preferences", "successful_strategies",
                        "failed_approaches", "learned_facts"]
        )

        if has_content:
            store.put(
                namespace=("learnings",),
                key=f"reflection_{timestamp}",
                value=learnings,
            )

            # Sauvegarde sur disque pour persistance entre les redémarrages
            save_memory_store(store)

    except Exception as e:
        # Si la réflexion échoue (LLM down, JSON invalide, etc.),
        # on continue silencieusement. La réflexion est un bonus,
        # pas une fonctionnalité critique.
        print(f"[Learning] Erreur lors de la réflexion : {e}")

    # Remise à zéro du compteur d'interactions
    return {**state, "interaction_count": 0}


# =============================================================================
# CONDITION : FAUT-IL DÉCLENCHER UNE RÉFLEXION ?
# =============================================================================

def should_reflect(state) -> str:
    """Détermine si l'agent doit déclencher une réflexion.

    Cette fonction est utilisée comme condition dans le graphe LangGraph
    pour router vers le noeud de réflexion ou directement vers la sortie.

    La réflexion est déclenchée quand le compteur d'interactions atteint
    REFLECTION_FREQUENCY (par défaut 5).

    Args:
        state: L'état actuel du graphe avec le compteur d'interactions.

    Returns:
        "reflect" si une réflexion est nécessaire, "skip" sinon.
    """
    count = state.get("interaction_count", 0)

    # On déclenche la réflexion tous les N échanges
    if count >= REFLECTION_FREQUENCY:
        return "reflect"
    else:
        return "skip"
