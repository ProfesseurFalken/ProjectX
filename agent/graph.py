"""
ProjectX - Graphe Agent LangGraph (Cœur du système)
Ce module définit le graphe d'exécution principal de l'agent AI.
Il orchestre la boucle de raisonnement ReAct enrichie avec mémoire
et auto-apprentissage.

Le graphe suit ce flux :
    1. RECALL   → Charge les apprentissages pertinents de la mémoire long-terme
    2. AGENT    → Le LLM raisonne et décide quelle action prendre
    3. TOOLS    → Exécute l'outil sélectionné par le LLM
    4. DECIDE   → Le LLM a-t-il fini ? Continue-t-il ? Faut-il réfléchir ?
    5. REFLECT  → (Périodique) Analyse les interactions, extrait des leçons
    6. → Retour à AGENT ou FIN

Diagramme du graphe :

    [START]
       ↓
    [recall] ──→ [agent] ←──────────────────────┐
                    ↓                            │
              tool_calls ?                       │
              ↓ OUI     ↓ NON                    │
           [tools]    [check_reflect]            │
              ↓           ↓ reflect   ↓ skip     │
              └──→ [agent] [reflect]  [END]      │
                              ↓                  │
                           [END]                 │

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

from typing import TypedDict, Annotated, Sequence, Optional, AsyncIterator
import operator

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, AIMessageChunk, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_MODEL_LIGHT,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS,
    SYSTEM_PROMPT,
)
from tools import ALL_TOOLS
from agent.memory import get_checkpointer, get_memory_store
from agent.learning import recall_memories, reflect_on_interactions, should_reflect
from agent.summarizer import should_summarize, summarize_conversation
from agent.planner import planner_node


# =============================================================================
# DÉFINITION DE L'ÉTAT DU GRAPHE
# =============================================================================

class AgentState(TypedDict):
    """État partagé entre tous les noeuds du graphe LangGraph.

    Chaque clé représente un "channel" du graphe. Les noeuds peuvent
    lire et écrire dans ces channels.

    Attributes:
        messages: Liste des messages de la conversation (user + assistant + tool).
                  Utilise operator.add comme réducteur pour que chaque noeud
                  puisse AJOUTER des messages sans écraser les précédents.
        recalled_memories: Texte des apprentissages rappelés de la mémoire
                          long-terme, injecté dans le prompt système.
        interaction_count: Compteur d'interactions pour déclencher la réflexion
                          périodique (tous les REFLECTION_FREQUENCY échanges).
    """
    # operator.add fait en sorte que les messages sont CONCATÉNÉS
    # (pas remplacés) quand un noeud retourne {"messages": [nouveau_msg]}
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Texte des apprentissages rappelés, injecté dans le system prompt
    recalled_memories: str

    # Compteur pour la réflexion périodique
    interaction_count: int


# =============================================================================
# NOEUDS DU GRAPHE
# =============================================================================

# Store de mémoire long-terme (singleton, partagé par tous les noeuds)
# Initialisé au premier appel de create_agent_graph()
_memory_store: Optional[InMemoryStore] = None


def _get_llm(model: str | None = None):
    """Crée et retourne une instance du LLM Ollama configurée avec les outils.

    Supporte le routage multi-modèles : par défaut utilise OLLAMA_MODEL
    (qwen2.5:14b), mais peut utiliser un modèle alternatif.

    Args:
        model: Nom du modèle Ollama à utiliser. Si None, utilise OLLAMA_MODEL.

    Returns:
        Instance ChatOllama avec les outils liés, prête à être invoquée.
    """
    llm = ChatOllama(
        model=model or OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_MAX_TOKENS,
    )

    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    return llm_with_tools


def _choose_model(state: dict) -> str:
    """Routage multi-modèles : choisit le modèle selon la complexité de la requête.

    Utilise le modèle léger (7b) pour les requêtes simples et courtes,
    et le modèle principal (14b) pour tout le reste.

    Critères pour le modèle léger :
    - Message court (< 100 caractères)
    - Pas de plan injecté (pas de tâche complexe détectée)
    - Pas de résultats d'outils en attente

    Returns:
        Le nom du modèle Ollama à utiliser.
    """
    if OLLAMA_MODEL == OLLAMA_MODEL_LIGHT:
        return OLLAMA_MODEL  # Routage désactivé

    messages = state.get("messages", [])
    if not messages:
        return OLLAMA_MODEL

    # Si un plan a été injecté, la tâche est complexe → modèle principal
    for msg in messages[-5:]:
        if isinstance(msg, SystemMessage) and "[PLAN D'EXÉCUTION]" in (msg.content or ""):
            return OLLAMA_MODEL

    # Si le dernier message est un ToolMessage, on est en boucle d'outils → principal
    from langchain_core.messages import ToolMessage
    if messages and isinstance(messages[-1], ToolMessage):
        return OLLAMA_MODEL

    # Trouver le dernier message utilisateur
    last_user = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user = msg.content
            break

    # Message court et simple → modèle léger
    if last_user and len(last_user) < 100:
        return OLLAMA_MODEL_LIGHT

    return OLLAMA_MODEL


def recall_node(state: AgentState) -> dict:
    """Noeud RECALL : rappelle les apprentissages pertinents de la mémoire.

    Ce noeud est le PREMIER exécuté dans le graphe. Il consulte la mémoire
    long-terme pour trouver les apprentissages en lien avec la requête
    actuelle de l'utilisateur.

    Args:
        state: L'état actuel du graphe avec les messages.

    Returns:
        L'état enrichi avec recalled_memories contenant les apprentissages
        pertinents sous forme de texte.
    """
    global _memory_store
    result = recall_memories(state, _memory_store)

    # On retourne seulement les champs modifiés
    return {"recalled_memories": result.get("recalled_memories", "")}


async def summarize_node(state: AgentState) -> dict:
    """Noeud SUMMARIZE : compresse l'historique si trop long.

    Vérifie si le nombre de messages dépasse SUMMARIZE_THRESHOLD.
    Si oui, résume les anciens messages en un SystemMessage et supprime
    les anciens via RemoveMessage. Sinon, passe sans modification.

    Args:
        state: L'état du graphe avec les messages.

    Returns:
        Dict avec les messages de suppression + résumé, ou dict vide.
    """
    state_dict: dict = dict(state)
    if not should_summarize(state_dict):
        return {}

    return await summarize_conversation(state_dict)


async def agent_node(state: AgentState) -> dict:
    """Noeud AGENT : le LLM raisonne et décide de l'action à prendre.

    C'est le noeud central du graphe. Il :
    1. Construit le prompt système avec les apprentissages rappelés
    2. Envoie tous les messages au LLM (historique complet)
    3. Reçoit la réponse du LLM (texte ou appel d'outil)
    4. Incrémente le compteur d'interactions

    Le noeud est async pour permettre l'utilisation de ainvoke() qui
    émet des événements de streaming capturables par astream_events().

    Args:
        state: L'état complet du graphe (messages, mémoire, compteur).

    Returns:
        L'état mis à jour avec la réponse du LLM ajoutée aux messages
        et le compteur incrémenté.
    """
    # --- Construction du prompt système ---
    # On insère les apprentissages rappelés dans le prompt
    recalled = state.get("recalled_memories", "")
    system_prompt = SYSTEM_PROMPT.format(recalled_memories=recalled)

    # --- Préparation des messages pour le LLM ---
    # On ajoute le prompt système en premier, puis l'historique complet
    messages = state["messages"]

    # Le prompt système est toujours le premier message
    all_messages = [SystemMessage(content=system_prompt)] + list(messages)

    # --- Routage multi-modèles ---
    # Choisit le modèle approprié selon la complexité de la requête
    model = _choose_model(dict(state))

    # --- Invocation async du LLM ---
    # ainvoke() permet au streaming via astream_events() de capturer
    # les tokens individuels émis par le LLM en temps réel
    llm = _get_llm(model)
    response = await llm.ainvoke(all_messages)

    # --- Mise à jour de l'état ---
    # On incrémente le compteur d'interactions (pour la réflexion)
    current_count = state.get("interaction_count", 0)

    return {
        "messages": [response],  # Ajouté aux messages existants via operator.add
        "interaction_count": current_count + 1,
    }


def reflection_node(state: AgentState) -> dict:
    """Noeud REFLECTION : analyse les interactions et extrait des leçons.

    Déclenché périodiquement selon REFLECTION_FREQUENCY. Appelle le module
    learning.py pour effectuer l'analyse et stocker les apprentissages.

    Args:
        state: L'état du graphe avec l'historique des messages.

    Returns:
        L'état avec le compteur d'interactions remis à zéro.
    """
    global _memory_store
    result = reflect_on_interactions(state, _memory_store)

    return {"interaction_count": result.get("interaction_count", 0)}


# =============================================================================
# LOGIQUE DE ROUTAGE (CONDITIONS)
# =============================================================================

def should_use_tools(state: AgentState) -> str:
    """Détermine si le LLM veut utiliser un outil ou répondre directement.

    Après le noeud AGENT, on vérifie si la réponse du LLM contient des
    appels d'outils (tool_calls). Si oui, on route vers TOOLS. Sinon,
    on vérifie s'il faut déclencher une réflexion.

    Args:
        state: L'état du graphe après le noeud AGENT.

    Returns:
        "tools" si le LLM veut utiliser un outil,
        "check_reflect" si le LLM a fini de répondre.
    """
    # Le dernier message est la réponse du LLM
    last_message = state["messages"][-1]

    # Si le message contient des tool_calls, le LLM veut utiliser un outil
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    else:
        # Le LLM a répondu directement — vérifier s'il faut réfléchir
        return "check_reflect"


def check_reflection_needed(state: AgentState) -> str:
    """Vérifie si une réflexion doit être déclenchée après la réponse.

    Args:
        state: L'état du graphe avec le compteur d'interactions.

    Returns:
        "reflect" pour déclencher la réflexion, "end" pour terminer.
    """
    return should_reflect(state)


# =============================================================================
# CONSTRUCTION DU GRAPHE
# =============================================================================

async def create_agent_graph() -> tuple:
    """Construit et compile le graphe LangGraph de l'agent.

    Cette fonction est le point d'entrée principal du module. Elle :
    1. Initialise la mémoire (checkpointer SQLite + store long-terme)
    2. Définit les noeuds du graphe (recall, agent, tools, reflection)
    3. Définit les transitions entre les noeuds
    4. Compile le graphe avec la persistance activée

    Cette fonction est async car le checkpointer AsyncSqliteSaver nécessite
    un event loop actif lors de son instanciation.

    Returns:
        Un tuple (graph, checkpointer, store) contenant :
        - graph : le graphe compilé, prêt à être invoqué
        - checkpointer : le AsyncSqliteSaver pour la persistance
        - store : l'InMemoryStore pour la mémoire long-terme
    """
    global _memory_store

    # --- Initialisation de la mémoire ---
    checkpointer = await get_checkpointer()
    _memory_store = get_memory_store()

    # --- Définition du graphe ---
    # StateGraph utilise AgentState comme structure d'état partagée
    builder = StateGraph(AgentState)

    # --- Ajout des noeuds ---
    # Chaque noeud est une fonction qui reçoit l'état et retourne
    # les modifications à appliquer à l'état

    # Noeud RECALL : rappel de la mémoire long-terme
    builder.add_node("recall", recall_node)

    # Noeud SUMMARIZE : compression de l'historique si nécessaire
    builder.add_node("summarize", summarize_node)

    # Noeud PLANNER : décomposition des tâches complexes
    builder.add_node("planner", planner_node)

    # Noeud AGENT : raisonnement du LLM (cœur de la boucle ReAct)
    builder.add_node("agent", agent_node)

    # Noeud TOOLS : exécution automatique des outils demandés par le LLM
    # ToolNode est un noeud pré-construit de LangGraph qui gère
    # automatiquement l'exécution des outils et la conversion des résultats
    tool_node = ToolNode(ALL_TOOLS)
    builder.add_node("tools", tool_node)

    # Noeud REFLECTION : analyse et apprentissage (périodique)
    builder.add_node("reflection", reflection_node)

    # --- Définition des transitions (edges) ---

    # Point d'entrée : on commence toujours par le rappel mémoire
    builder.set_entry_point("recall")

    # recall → summarize : après le rappel, compresser si nécessaire
    builder.add_edge("recall", "summarize")

    # summarize → planner : après compression, analyser la complexité
    builder.add_edge("summarize", "planner")

    # planner → agent : après planification, passer au raisonnement
    builder.add_edge("planner", "agent")

    # agent → tools OU check_reflect : selon que le LLM veut un outil ou non
    builder.add_conditional_edges(
        "agent",
        should_use_tools,
        {
            "tools": "tools",                # Le LLM veut un outil → exécuter
            "check_reflect": "check_reflect", # Le LLM a répondu → vérifier réflexion
        },
    )

    # tools → agent : après exécution de l'outil, retour au LLM
    # pour qu'il interprète le résultat
    builder.add_edge("tools", "agent")

    # Noeud virtuel de vérification de réflexion
    # On utilise un conditional edge depuis un "point de décision"
    # IMPORTANT : retourner {} et NON state — sinon operator.add sur
    # messages re-ajouterait tous les messages existants, les doublant
    builder.add_node("check_reflect", lambda state: {})  # Passthrough
    builder.add_conditional_edges(
        "check_reflect",
        check_reflection_needed,
        {
            "reflect": "reflection",  # Réflexion nécessaire
            "skip": END,              # Pas de réflexion, fin du tour
        },
    )

    # reflection → END : après la réflexion, le tour est fini
    builder.add_edge("reflection", END)

    # --- Compilation du graphe ---
    # Le checkpointer assure la persistance des conversations
    # Le store est accessible via les closures des noeuds
    # recursion_limit empêche les boucles infinies agent↔tools.
    # 25 itérations = ~12 appels d'outils maximum, largement suffisant
    # pour la plupart des tâches.
    graph = builder.compile(checkpointer=checkpointer)

    return graph, checkpointer, _memory_store
