"""
ProjectX - Planificateur de Tâches Complexes
Détecte les requêtes multi-étapes et décompose le travail en sous-tâches
avant de les exécuter. Cela permet à Joshua de mieux structurer ses
réponses sur des demandes complexes.

Le planificateur intervient juste après le recall et avant le noeud agent.
Il analyse la requête de l'utilisateur et, si elle est complexe, injecte
un plan d'exécution sous forme de SystemMessage que l'agent suivra.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_LIGHT,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent.graph import AgentState


# Prompt pour analyser la complexité et planifier
_PLANNING_PROMPT = """Tu es un analyseur de tâches. Tu reçois la requête d'un utilisateur et tu dois décider si elle est SIMPLE ou COMPLEXE.

Une tâche est COMPLEXE si elle nécessite AU MOINS 2 de ces critères :
- Plusieurs recherches web distinctes
- Combiner des informations de sources multiples
- Écrire du code ET l'exécuter
- Créer ou modifier plusieurs fichiers
- Comparer ou analyser des données
- Effectuer des actions en séquence
- Un rapport ou une synthèse de recherche long

Si la tâche est SIMPLE, réponds uniquement : SIMPLE

Si la tâche est COMPLEXE, réponds avec un plan structuré :
COMPLEXE
PLAN:
1. [première étape]
2. [deuxième étape]
3. [troisième étape]
...

Pas plus de 6 étapes. Sois concis (une ligne par étape).

Requête utilisateur :
{query}"""


def _get_last_user_message(state: dict) -> str:
    """Extrait le dernier message de l'utilisateur."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


async def planner_node(state: "AgentState") -> dict:
    """Noeud PLANNER : analyse la complexité et injecte un plan si nécessaire.

    Ce noeud est exécuté après recall et avant agent. Il utilise le modèle
    léger (qwen2.5:7b) pour rapidement classifier la requête :
    - SIMPLE → passe directement (retourne {} sans modification)
    - COMPLEXE → injecte un SystemMessage avec le plan d'exécution

    Le plan guide ensuite le noeud agent dans sa stratégie de résolution.

    Args:
        state: L'état du graphe avec les messages.

    Returns:
        Dict vide si la tâche est simple, ou dict avec un SystemMessage
        contenant le plan pour les tâches complexes.
    """
    query = _get_last_user_message(dict(state))  # type: ignore[arg-type]

    if not query:
        return {}

    # Détection des demandes de liberté/exploration
    # L'utilisateur veut que l'agent agisse de manière autonome
    _FREEDOM_KEYWORDS = [
        "fais ce que tu veux", "libre de", "choisis toi", "choisis un sujet",
        "explore internet", "consulte internet", "navigue sur", "apprends",
        "améliore-toi", "ameliore-toi", "sois autonome", "prends l'initiative",
        "fais le toi", "fais-le toi", "à toi de jouer", "a toi de jouer",
        "ce que tu souhaites", "ce que tu veux", "comme tu veux",
    ]
    query_lower = query.lower()
    if any(kw in query_lower for kw in _FREEDOM_KEYWORDS):
        return {
            "messages": [SystemMessage(content=(
                "[DIRECTIVE] L'utilisateur te donne carte blanche. "
                "Tu DOIS agir MAINTENANT sans poser de question. "
                "Étape 1 : Appelle web_search avec un sujet de TON choix (actualité tech, science, découverte récente...). "
                "Étape 2 : Appelle scrape_webpage sur le meilleur résultat. "
                "Étape 3 : Fais une synthèse passionnante de ce que tu as lu. "
                "NE DEMANDE PAS L'AVIS de l'utilisateur. AGIS."
            ))]
        }

    # Si le message est court ou vide, pas besoin de planifier
    if len(query) < 80:
        return {}

    # Utiliser le modèle léger pour la classification rapide
    llm = ChatOllama(
        model=OLLAMA_MODEL_LIGHT,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=300,
    )

    prompt = _PLANNING_PROMPT.format(query=query)

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content
        content = raw.strip() if isinstance(raw, str) else str(raw).strip()

        # Si la tâche est simple, pas de plan nécessaire
        if content.startswith("SIMPLE"):
            return {}

        # Extraire le plan
        if "PLAN:" in content:
            plan_text = content.split("PLAN:", 1)[1].strip()
            plan_message = SystemMessage(
                content=(
                    f"[PLAN D'EXÉCUTION]\n"
                    f"Cette requête est complexe. Suis ce plan étape par étape :\n"
                    f"{plan_text}\n"
                    f"Exécute chaque étape dans l'ordre, utilise les outils nécessaires, "
                    f"puis fais une synthèse finale."
                )
            )
            return {"messages": [plan_message]}

    except Exception:
        # En cas d'erreur du planificateur, on continue sans plan
        pass

    return {}
