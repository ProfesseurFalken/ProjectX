"""
ProjectX - Résumé Automatique de Conversation
Quand l'historique de messages dépasse un seuil (SUMMARIZE_THRESHOLD),
les anciens messages sont compressés en un résumé concis. Cela empêche
le dépassement de la fenêtre de contexte du LLM (32K tokens pour qwen2.5).

Seuls les SUMMARIZE_KEEP_RECENT messages les plus récents sont gardés
intacts. Le reste est remplacé par un unique SystemMessage contenant
le résumé des échanges passés.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage,
)
from langchain_ollama import ChatOllama

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_LIGHT,
    SUMMARIZE_THRESHOLD,
    SUMMARIZE_KEEP_RECENT,
)


# Prompt utilisé pour demander au LLM de résumer la conversation
_SUMMARIZE_PROMPT = """Tu es un système de compression de conversation.
Résume la conversation suivante de manière concise mais complète.
Conserve les informations clés :
- Les sujets abordés et les questions de l'utilisateur
- Les résultats importants trouvés (recherches web, fichiers lus, etc.)
- Les décisions prises et les actions effectuées
- Les préférences de l'utilisateur détectées

Écris le résumé en français, au format narratif concis (pas de bullet points).
Maximum 500 mots.

Conversation à résumer :
{conversation}"""


def should_summarize(state: dict) -> bool:
    """Vérifie si l'historique doit être compressé.

    Args:
        state: L'état du graphe contenant les messages.

    Returns:
        True si le nombre de messages dépasse SUMMARIZE_THRESHOLD.
    """
    messages = state.get("messages", [])
    return len(messages) > SUMMARIZE_THRESHOLD


async def summarize_conversation(state: dict) -> dict:
    """Résume les anciens messages et retourne les messages à supprimer/ajouter.

    Garde les SUMMARIZE_KEEP_RECENT messages les plus récents intacts.
    Les anciens messages sont résumés par le LLM léger, puis remplacés
    par un unique SystemMessage contenant le résumé.

    Args:
        state: L'état du graphe contenant les messages.

    Returns:
        Dict avec "messages" contenant : RemoveMessage pour chaque ancien
        message + un SystemMessage avec le résumé.
    """
    messages = state.get("messages", [])

    if len(messages) <= SUMMARIZE_THRESHOLD:
        return {}

    # --- Séparation : anciens messages vs messages récents ---
    old_messages = messages[:-SUMMARIZE_KEEP_RECENT]
    # Les messages récents sont gardés par le graphe via operator.add

    # --- Construction du texte à résumer ---
    conversation_text = ""
    for msg in old_messages:
        if isinstance(msg, HumanMessage):
            conversation_text += f"UTILISATEUR : {msg.content}\n"
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content and len(content) > 300:
                content = content[:300] + "..."
            if content:
                conversation_text += f"ASSISTANT : {content}\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "outil")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 200:
                content = content[:200] + "..."
            conversation_text += f"OUTIL [{tool_name}] : {content}\n"

    if not conversation_text.strip():
        return {}

    # --- Appel au LLM léger pour résumer ---
    try:
        summary_llm = ChatOllama(
            model=OLLAMA_MODEL_LIGHT,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
        prompt = _SUMMARIZE_PROMPT.format(conversation=conversation_text)
        response = await summary_llm.ainvoke([HumanMessage(content=prompt)])
        summary_text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        # En cas d'échec, résumé basique par troncature
        summary_text = f"[Résumé des {len(old_messages)} premiers messages — résumé automatique indisponible]"

    # --- Construction des messages de remplacement ---
    # On utilise RemoveMessage pour supprimer les anciens messages du checkpoint
    # puis on insère un SystemMessage avec le résumé
    result_messages = []

    # Suppression des anciens messages
    for msg in old_messages:
        if hasattr(msg, "id") and msg.id:
            result_messages.append(RemoveMessage(id=msg.id))

    # Insertion du résumé en tant que SystemMessage
    summary_msg = SystemMessage(
        content=f"[RÉSUMÉ DE LA CONVERSATION PRÉCÉDENTE]\n{summary_text}\n[FIN DU RÉSUMÉ]"
    )
    result_messages.append(summary_msg)

    return {"messages": result_messages}
