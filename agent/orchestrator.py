"""
ProjectX - Orchestrateur Multi-Agents
Analyse la requête utilisateur et route vers le spécialiste approprié.
Utilise un routage hybride : heuristique rapide + LLM en fallback.

L'orchestrateur remplace le planner dans le graphe LangGraph :
    recall → summarize → orchestrator → agent_specialist → tools → ...

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-29
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL_LIGHT
from agent.specialists import ROUTING_KEYWORDS, SPECIALISTS

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent.graph import AgentState


# Mots-clés de liberté/exploration → ResearchAgent par défaut
_FREEDOM_KEYWORDS = [
    "fais ce que tu veux", "libre de", "choisis toi", "choisis un sujet",
    "explore internet", "consulte internet", "navigue sur", "apprends",
    "améliore-toi", "ameliore-toi", "sois autonome", "prends l'initiative",
    "fais le toi", "fais-le toi", "à toi de jouer", "a toi de jouer",
    "ce que tu souhaites", "ce que tu veux", "comme tu veux",
    "je te laisse", "décide toi", "decide toi",
    "maniere autonome", "manière autonome", "toute ma confiance",
    "tu as carte blanche", "carte blanche", "fais le de",
    "agis librement", "totale liberté", "totale liberte",
    "surprends-moi", "surprends moi", "étonne-moi", "etonne-moi",
    "tu decides", "tu décides", "choisis pour moi",
]

# Prompt pour le routage LLM (utilisé quand l'heuristique n'est pas confiante)
_ROUTING_PROMPT = """Tu es un routeur. Analyse cette requête et réponds avec UN SEUL mot parmi : research, coder, system, memory, general.

- research : recherche web, actualités, information, navigation internet
- coder : code, programmation, calcul, script, données
- system : commandes système, fichiers, dossiers, email, installation
- memory : souvenirs, rappels, mémorisation, documents indexés
- general : conversation, questions simples, salutations, tout le reste

Requête : {query}

Réponse (un seul mot) :"""


def _get_last_user_message(state: dict) -> str:
    """Extrait le dernier message de l'utilisateur."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _heuristic_route(query: str) -> tuple[str, float]:
    """Routage heuristique par mots-clés. Retourne (spécialiste, confiance).

    La confiance est entre 0.0 et 1.0. Si > 0.6, on utilise le résultat
    directement sans appel LLM.
    """
    query_lower = query.lower()

    # Vérifier les mots-clés de liberté → research
    if any(kw in query_lower for kw in _FREEDOM_KEYWORDS):
        return "research", 1.0

    # Compter les matchs par spécialiste
    scores: dict[str, int] = {}
    for specialist, keywords in ROUTING_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[specialist] = score

    if not scores:
        return "general", 0.3  # Pas de match → general avec faible confiance

    # Le spécialiste avec le plus de matchs
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    total_keywords = sum(scores.values())
    confidence = min(scores[best] / max(total_keywords, 1), 1.0)

    # Si le meilleur a au moins 2 matchs ET domine clairement
    if scores[best] >= 2 and confidence >= 0.5:
        return best, confidence

    return best, confidence * 0.7  # Réduire la confiance si peu de matchs


async def _llm_route(query: str) -> str:
    """Routage par LLM léger. Utilisé quand l'heuristique est incertaine."""
    llm = ChatOllama(
        model=OLLAMA_MODEL_LIGHT,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=20,
    )
    prompt = _ROUTING_PROMPT.format(query=query)
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content
        result = raw.strip().lower() if isinstance(raw, str) else str(raw).strip().lower()
        # Extraire le premier mot valide
        for word in result.split():
            clean = word.strip(".,;:!?\"'")
            if clean in SPECIALISTS:
                return clean
    except Exception:
        pass
    return "general"


async def orchestrator_node(state: "AgentState") -> dict:
    """Noeud ORCHESTRATEUR : route la requête vers le bon spécialiste.

    Stratégie hybride :
    1. Heuristique par mots-clés (instantané, 0ms)
    2. Si confiance < 0.6 → fallback LLM léger (~200ms)

    Injecte dans l'état :
    - current_specialist : nom du spécialiste sélectionné
    - Un SystemMessage de directive si c'est une demande de liberté

    Args:
        state: L'état du graphe.

    Returns:
        Dict avec current_specialist + éventuellement un SystemMessage.
    """
    query = _get_last_user_message(dict(state))

    if not query:
        return {"current_specialist": "general"}

    # Étape 1 : Heuristique rapide
    specialist, confidence = _heuristic_route(query)

    # Étape 2 : Si pas confiant, demander au LLM
    if confidence < 0.6:
        specialist = await _llm_route(query)

    import logging
    logging.getLogger(__name__).info(
        f"Orchestrator: '{query[:60]}...' → {specialist} (confiance={confidence:.2f})"
    )

    result: dict = {"current_specialist": specialist}

    # Injection de directive pour les demandes de liberté
    query_lower = query.lower()
    if any(kw in query_lower for kw in _FREEDOM_KEYWORDS):
        result["messages"] = [SystemMessage(content=(
            "[DIRECTIVE] L'utilisateur te donne carte blanche. "
            "Tu DOIS agir MAINTENANT sans poser de question. "
            "Étape 1 : Appelle web_search avec un sujet de TON choix "
            "(actualité tech, science, découverte récente...). "
            "Étape 2 : Appelle scrape_webpage sur le meilleur résultat. "
            "Étape 3 : Fais une synthèse passionnante de ce que tu as lu. "
            "NE DEMANDE PAS L'AVIS de l'utilisateur. AGIS."
        ))]

    return result
