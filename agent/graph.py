"""
ProjectX - Graphe Agent LangGraph Multi-Agents (Cœur du système)
Ce module définit le graphe d'exécution principal de l'agent AI.
Il orchestre une architecture multi-agents avec routage intelligent
vers des spécialistes (Research, Coder, System, Memory).

Le graphe suit ce flux :
    1. RECALL        → Charge les apprentissages pertinents
    2. SUMMARIZE     → Compresse l'historique si trop long
    3. ORCHESTRATOR  → Route vers le spécialiste approprié
    4. AGENT         → Le spécialiste raisonne avec ses outils dédiés
    5. TOOLS         → Exécute l'outil sélectionné
    6. CHECK_REFLECT → Faut-il réfléchir ?
    7. REFLECT       → (Périodique) Extrait des leçons
    8. → Retour à AGENT ou FIN

Diagramme du graphe :

    [START]
       ↓
    [recall] → [summarize] → [orchestrator] → [agent] ←────┐
                                                 ↓          │
                                           tool_calls ?     │
                                          ↓ OUI   ↓ NON    │
                                       [tools] [check_reflect]
                                          ↓      ↓ reflect  ↓ skip
                                          └─→ [agent] [reflect]  [END]
                                                          ↓
                                                       [END]

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-29
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
from agent.orchestrator import orchestrator_node
from agent.specialists import SPECIALISTS


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

    # Spécialiste sélectionné par l'orchestrateur pour ce tour
    current_specialist: str


# =============================================================================
# NOEUDS DU GRAPHE
# =============================================================================

# Store de mémoire long-terme (singleton, partagé par tous les noeuds)
# Initialisé au premier appel de create_agent_graph()
_memory_store: Optional[InMemoryStore] = None


def _get_llm(model: str | None = None, tools: list | None = None):
    """Crée et retourne une instance du LLM Ollama configurée avec les outils.

    Supporte le routage multi-modèles et les outils spécialisés.

    Args:
        model: Nom du modèle Ollama à utiliser. Si None, utilise OLLAMA_MODEL.
        tools: Liste d'outils à lier au LLM. Si None, utilise ALL_TOOLS.

    Returns:
        Instance ChatOllama avec les outils liés, prête à être invoquée.
    """
    llm = ChatOllama(
        model=model or OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_MAX_TOKENS,
        repeat_penalty=1.2,
        reasoning=False,  # Désactiver le mode thinking de qwen3 pour l'agent
    )

    bound_tools = tools if tools is not None else ALL_TOOLS
    llm_with_tools = llm.bind_tools(bound_tools)

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

    Utilise le spécialiste sélectionné par l'orchestrateur pour réduire
    le nombre d'outils visibles par le LLM, améliorant la précision
    du tool-calling.

    Args:
        state: L'état complet du graphe (messages, mémoire, compteur, spécialiste).

    Returns:
        L'état mis à jour avec la réponse du LLM ajoutée aux messages
        et le compteur incrémenté.
    """
    from langchain_core.messages import ToolMessage
    from tools.task_status import task_start, task_end

    # --- Sélection du spécialiste ---
    specialist_key = state.get("current_specialist", "general")
    specialist = SPECIALISTS.get(specialist_key, SPECIALISTS["general"])
    specialist_tools = specialist["tools"]  # None = ALL_TOOLS
    specialist_prompt = specialist["prompt"]

    import logging
    _agent_logger = logging.getLogger(__name__)
    tool_count = len(specialist_tools) if specialist_tools else len(ALL_TOOLS)
    _agent_logger.info(
        f"Agent: spécialiste={specialist['name']}, outils={tool_count}"
    )

    # --- Construction du prompt système ---
    recalled = state.get("recalled_memories", "")
    base_prompt = SYSTEM_PROMPT.format(recalled_memories=recalled)
    # Injecter la directive du spécialiste
    system_prompt = f"{base_prompt}\n\n[RÔLE ACTIF : {specialist['name']}]\n{specialist_prompt}"

    # --- Préparation des messages pour le LLM ---
    raw_messages = list(state["messages"])

    # Absorber les SystemMessages de directive (orchestrateur) dans le system prompt
    # au lieu de les laisser comme messages séparés
    directives = []
    user_messages = []
    for m in raw_messages:
        if isinstance(m, SystemMessage) and "[DIRECTIVE]" in (m.content or ""):
            directives.append(m.content)
        else:
            user_messages.append(m)
    if directives:
        system_prompt += "\n\n" + "\n".join(directives)

    # FILTRE 1 : Ne garder que les types supportés par Ollama
    supported_types = (SystemMessage, HumanMessage, AIMessage, ToolMessage)
    filtered = [m for m in user_messages if isinstance(m, supported_types)]

    # FILTRE 2 : Limiter le contexte (30 messages max)
    MAX_CONTEXT_MESSAGES = 30
    if len(filtered) > MAX_CONTEXT_MESSAGES:
        filtered = filtered[-MAX_CONTEXT_MESSAGES:]
        while filtered and isinstance(filtered[0], ToolMessage):
            filtered = filtered[1:]

    # Un seul SystemMessage en tête — pas de SystemMessage isolés dans le corps
    all_messages = [SystemMessage(content=system_prompt)] + filtered

    # --- Invocation async du LLM avec outils du spécialiste ---
    # Extraire un résumé du dernier message humain pour le task tracking
    _last_user = ""
    for _m in reversed(filtered):
        if isinstance(_m, HumanMessage):
            _last_user = (_m.content if isinstance(_m.content, str) else str(_m.content))[:80]
            break
    task_start(f"{specialist['name']}", detail=_last_user)

    llm = _get_llm(tools=specialist_tools)
    response = await llm.ainvoke(all_messages)

    # --- Anti-passivité : détecter refus ou questions au lieu d'agir ---
    # EXCEPTION : pour les salutations simples, une réponse textuelle est NORMALE
    _GREETING_PATTERNS = [
        "bonjour", "salut", "hello", "hey", "coucou", "bonsoir",
        "hi joshua", "bonjour joshua", "salut joshua",
        "comment vas-tu", "comment tu vas", "ça va",
    ]
    _last_user_msg = ""
    for _m in reversed(list(state["messages"])):
        if isinstance(_m, HumanMessage):
            _last_user_msg = (_m.content if isinstance(_m.content, str) else str(_m.content)).lower().strip()
            break
    _is_greeting = any(g in _last_user_msg for g in _GREETING_PATTERNS) and len(_last_user_msg) < 80

    if (isinstance(response, AIMessage)
            and response.content
            and not response.tool_calls
            and not _is_greeting):
        text = response.content if isinstance(response.content, str) else str(response.content)
        text_lower = text.lower()

        # Log la réponse textuelle complète pour diagnostic
        _agent_logger.info(
            f"Agent: réponse texte de {specialist['name']} ({len(text)} car): "
            f"{text[:200]}{'...' if len(text) > 200 else ''}"
        )

        # Détection de texte non-français (hébreu, arabe, chinois, etc.)
        import re
        _non_latin_ratio = len(re.findall(r'[^\x00-\x7F\xC0-\xFF]', text)) / max(len(text), 1)
        if _non_latin_ratio > 0.3:
            _agent_logger.warning(
                f"Agent: texte non-latin détecté ({_non_latin_ratio:.0%}), régénération en français."
            )
            # Ajouter une instruction de langue et ré-invoquer
            all_messages.append(AIMessage(content=text))
            all_messages.append(HumanMessage(content="IMPORTANT: Réponds UNIQUEMENT en français. Reformule ta réponse précédente en français."))
            response = await llm.ainvoke(all_messages)

        # Patterns de passivité (questions) ET de refus
        _PASSIVE_PATTERNS = [
            "que dirais-tu", "comment te semble", "voulez-vous que",
            "si vous souhaitez", "quel sujet", "qu'en penses-tu",
            "pourriez-vous", "pouvez-vous me préciser", "souhaitez-vous",
            "que souhaitez", "puis-je vous aider", "que puis-je faire",
            "n'hésitez pas", "veuillez préciser",
            # Patterns de refus
            "je ne peux pas", "je suis désolé", "je suis desole",
            "pas en mesure", "mes limites", "limitations",
            "pas développer de nouveaux outils",
            "sans autorisation", "sans interfaces",
            "je ne suis pas capable", "il m'est impossible",
            "pas autorisé", "pas possible pour moi",
            "sécurité et de confidentialité",
            "pour toute autre demande",
        ]

        is_passive = any(p in text_lower for p in _PASSIVE_PATTERNS)

        if is_passive:
            import logging
            logging.getLogger(__name__).warning(
                f"Anti-passivité: réponse passive détectée du {specialist_key}Agent. "
                f"Forçage d'un tool_call adapté au spécialiste."
            )
            # Forcer un tool_call adapté au spécialiste actif
            from langchain_core.messages import AIMessage as _AIMessage
            _FORCED_ACTIONS = {
                "research": {"name": "web_search", "args": {"query": "dernières découvertes scientifiques 2026"}},
                "coder": {"name": "list_directory_tree", "args": {"dir_path": ".", "max_depth": 3}},
                "system": {"name": "run_command", "args": {"command": "echo Systeme pret"}},
                "memory": {"name": "recall_memory", "args": {"query": "derniers souvenirs"}},
                "general": {"name": "web_search", "args": {"query": "dernières découvertes scientifiques 2026"}},
            }
            action = _FORCED_ACTIONS.get(specialist_key, _FORCED_ACTIONS["general"])
            forced_response = _AIMessage(
                content="",
                tool_calls=[{
                    "id": "forced_action_001",
                    "name": action["name"],
                    "args": action["args"],
                }],
            )
            response = forced_response

    # --- Tracking de fin de tâche ---
    if isinstance(response, AIMessage) and response.tool_calls:
        tool_names = ", ".join(tc["name"] for tc in response.tool_calls)
        task_end(f"{specialist['name']}", result=f"→ {tool_names}")
    else:
        task_end(f"{specialist['name']}", result="réponse texte")

    # --- Mise à jour de l'état ---
    current_count = state.get("interaction_count", 0)

    return {
        "messages": [response],
        "interaction_count": current_count + 1,
    }


async def reflection_node(state: AgentState) -> dict:
    """Noeud REFLECTION : analyse les interactions et extrait des leçons.

    Déclenché périodiquement selon REFLECTION_FREQUENCY. Appelle le module
    learning.py pour effectuer l'analyse et stocker les apprentissages.

    Args:
        state: L'état du graphe avec l'historique des messages.

    Returns:
        L'état avec le compteur d'interactions remis à zéro.
    """
    global _memory_store
    result = await reflect_on_interactions(state, _memory_store)

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

    # Noeud ORCHESTRATOR : route vers le spécialiste approprié
    builder.add_node("orchestrator", orchestrator_node)

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

    # summarize → orchestrator : après compression, router vers le spécialiste
    builder.add_edge("summarize", "orchestrator")

    # orchestrator → agent : après routage, passer au raisonnement
    builder.add_edge("orchestrator", "agent")

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
