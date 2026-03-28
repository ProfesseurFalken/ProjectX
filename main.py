"""
ProjectX - Point d'Entrée Principal (Interface Web Chainlit)
Ce fichier est le point d'entrée de l'application. Il configure et lance
l'interface web Chainlit qui permet à l'utilisateur d'interagir avec
l'agent AI via un chat dans le navigateur.

Chainlit gère :
- L'affichage du chat (messages utilisateur + réponses de l'agent)
- Le streaming des réponses en temps réel
- L'affichage des étapes intermédiaires (outils utilisés, résultats)
- La gestion des sessions utilisateur

Lancement : `chainlit run main.py` → ouvre http://localhost:8000

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import uuid

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from agent.graph import create_agent_graph
from agent.sessions import save_session, get_recent_sessions

# Type pour l'annotation du graphe compilé
from langgraph.graph.state import CompiledStateGraph

# =============================================================================
# VARIABLES GLOBALES DE SESSION
# =============================================================================

# Le graphe de l'agent, le checkpointer et le store sont créés une seule fois
# au démarrage du serveur Chainlit et partagés entre toutes les sessions.
# Chaque utilisateur a son propre thread_id pour isoler ses conversations.
_graph: CompiledStateGraph | None = None
_checkpointer = None
_store = None


# =============================================================================
# ÉVÉNEMENT : DÉMARRAGE DU CHAT
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Appelé quand un nouvel utilisateur ouvre le chat.

    Initialise le graphe de l'agent (si pas encore fait) et crée un
    identifiant unique pour cette conversation (thread_id). Le thread_id
    permet au système de persistance (SqliteSaver) d'identifier et de
    stocker l'historique de cette conversation spécifique.
    """
    global _graph, _checkpointer, _store

    # --- Initialisation du graphe (une seule fois, premier utilisateur) ---
    if _graph is None:
        # Envoi d'un message de chargement pendant l'initialisation
        # (le premier chargement peut prendre quelques secondes)
        msg = cl.Message(content="Initialisation de Joshua...")
        await msg.send()

        # Création du graphe LangGraph avec tous les outils et la mémoire
        # create_agent_graph est async car le checkpointer nécessite un event loop
        _graph, _checkpointer, _store = await create_agent_graph()

        # Mise à jour du message de chargement
        msg.content = (
            "**Joshua** est prêt !\n\n"
            "Je suis ton assistant AI autonome. Je peux :\n"
            "- Chercher des informations sur le web et les lire\n"
            "- Naviguer dans un navigateur automatisé\n"
            "- Gérer tes fichiers (lire, écrire, déplacer, supprimer)\n"
            "- Exécuter des commandes système\n"
            "- Envoyer des emails\n"
            "- Écrire et exécuter du code Python\n\n"
            "Que puis-je faire pour toi ?"
        )
        await msg.update()
    else:
        # L'agent est déjà initialisé — message d'accueil simple
        await cl.Message(
            content=(
                "**Joshua** — Agent AI autonome prêt.\n"
                "Comment puis-je t'aider ?"
            )
        ).send()

    # --- Création d'un thread_id unique pour cette conversation ---
    # Le thread_id est stocké dans la session Chainlit de l'utilisateur
    # Il est utilisé par le checkpointer pour persister les messages
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)

    # --- Afficher les sessions récentes si disponibles ---
    recent = get_recent_sessions(limit=5)
    if recent:
        lines = ["**Sessions précédentes :**"]
        for i, s in enumerate(recent, 1):
            title = s.get("title", "Sans titre")
            date = s.get("last_active", "")[:16].replace("T", " ")
            lines.append(f"{i}. _{title}_ — {date}")
        lines.append(
            "\n_Pour reprendre une session, tape son numéro (ex: `1`)._"
        )
        await cl.Message(content="\n".join(lines)).send()

    cl.user_session.set("first_message", True)


# =============================================================================
# ÉVÉNEMENT : NOUVEAU MESSAGE DE L'UTILISATEUR
# =============================================================================

@cl.on_message
async def on_message(message: cl.Message):
    """Appelé quand l'utilisateur envoie un message dans le chat.

    Transmet le message au graphe LangGraph, suit l'exécution des outils
    en temps réel, et affiche la réponse finale à l'utilisateur.

    Le flux est :
    1. Message utilisateur → graphe LangGraph
    2. Le graphe exécute recall → agent → tools → agent → ...
    3. Les étapes intermédiaires sont affichées comme "Steps" Chainlit
    4. La réponse finale de l'agent est affichée dans le chat

    Args:
        message: L'objet Message Chainlit contenant le texte de l'utilisateur.
    """
    global _graph, _store

    # Récupération du thread_id de la session courante
    thread_id = cl.user_session.get("thread_id")

    # --- Reprise de session par numéro ---
    is_first = cl.user_session.get("first_message", False)
    if is_first and message.content.strip().isdigit():
        idx = int(message.content.strip()) - 1
        recent = get_recent_sessions(limit=5)
        if 0 <= idx < len(recent):
            old_session = recent[idx]
            cl.user_session.set("thread_id", old_session["thread_id"])
            thread_id = old_session["thread_id"]
            cl.user_session.set("first_message", False)
            await cl.Message(
                content=f"Session reprise : _{old_session.get('title', 'Sans titre')}_\n"
                        f"Continuons la conversation !"
            ).send()
            return

    cl.user_session.set("first_message", False)

    # --- Sauvegarder la session ---
    if thread_id:
        save_session(thread_id, message.content)

    # Vérification que le graphe est initialisé (toujours vrai si on_chat_start a été appelé)
    if _graph is None:
        await cl.Message(content="Erreur : l'agent n'est pas encore initialisé.").send()
        return

    # --- Configuration du graphe pour cette invocation ---
    # Le thread_id permet au checkpointer de retrouver/sauvegarder
    # l'historique de cette conversation spécifique
    # recursion_limit empêche les boucles infinies agent↔tools
    # 50 étapes permet ~20 appels d'outils : assez pour search + scrape
    # de 2-3 pages + synthèse, tout en protégeant contre les boucles
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 50,
    }

    # --- Construction de l'entrée pour le graphe ---
    # On envoie le message de l'utilisateur. Le recalled_memories sera
    # rempli par le noeud recall. On ne passe PAS interaction_count ici
    # car c'est un champ sans réducteur : le passer ici écraserait la
    # valeur persistée dans le checkpoint, empêchant la réflexion.
    input_data = {
        "messages": [HumanMessage(content=message.content)],
        "recalled_memories": "",  # Sera rempli par le noeud recall
    }

    # --- Exécution du graphe avec streaming token par token ---
    # On utilise astream_events(version="v2") qui capture TOUS les événements
    # du graphe, y compris les tokens individuels émis par le LLM (on_chat_model_stream).
    # Cela permet d'afficher la réponse progressivement dans Chainlit,
    # au lieu d'attendre que le LLM ait fini de générer toute sa réponse.
    final_response = ""
    pending_tool_calls = {}  # Mémorise les appels d'outils pour l'affichage
    streaming_msg = None     # Message Chainlit pour le streaming token par token
    is_streaming_text = False  # True quand on streame la réponse textuelle finale

    try:
        async for event in _graph.astream_events(
            input_data, config, version="v2"
        ):
            event_kind = event.get("event", "")
            tags = event.get("tags", [])

            # --- STREAMING TOKEN PAR TOKEN ---
            # Les événements on_chat_model_stream sont émis pour chaque token
            # généré par le LLM dans le noeud "agent"
            if event_kind == "on_chat_model_stream" and "agent" in tags:
                chunk = event.get("data", {}).get("chunk")
                if chunk is not None:
                    # Le chunk est un AIMessageChunk avec du contenu textuel
                    token = ""
                    if hasattr(chunk, "content") and isinstance(chunk.content, str):
                        token = chunk.content

                    # On ne streame que le texte (pas les tool_calls)
                    if token and not getattr(chunk, "tool_calls", None):
                        if streaming_msg is None:
                            # Premier token → créer le message Chainlit
                            streaming_msg = cl.Message(content="")
                            await streaming_msg.send()
                            is_streaming_text = True

                        # Ajout du token au message en streaming
                        await streaming_msg.stream_token(token)
                        final_response += token

            # --- FIN DU STREAMING D'UN MESSAGE COMPLET ---
            elif event_kind == "on_chat_model_end" and "agent" in tags:
                output = event.get("data", {}).get("output")
                if output is not None and hasattr(output, "tool_calls") and output.tool_calls:
                    # Le LLM veut utiliser des outils → mémoriser les appels
                    for tc in output.tool_calls:
                        pending_tool_calls[tc["id"]] = tc

                    # Si on était en train de streamer du texte avant un tool_call,
                    # on finalise le message (cas rare mais possible)
                    if streaming_msg and is_streaming_text:
                        await streaming_msg.update()
                        streaming_msg = None
                        is_streaming_text = False
                        final_response = ""  # Reset car ce n'est pas la réponse finale

                elif streaming_msg and is_streaming_text:
                    # Fin du streaming de la réponse textuelle
                    # Filtrer les réponses JSON (réflexion interne qui fuite)
                    stripped = final_response.strip()
                    if stripped.startswith("{") and stripped.endswith("}"):
                        # C'est du JSON interne → supprimer ce message
                        streaming_msg.content = ""
                        await streaming_msg.update()
                        streaming_msg = None
                        is_streaming_text = False
                        final_response = ""
                    else:
                        await streaming_msg.update()
                        # On garde streaming_msg et final_response pour la fin

            # --- RÉSULTATS DES OUTILS ---
            elif event_kind == "on_tool_end":
                output = event.get("data", {}).get("output")
                tool_name = event.get("name", "outil")

                if output is not None:
                    current_step = cl.Step(name=f"🔧 {tool_name}", type="tool")

                    # Récupération de l'input depuis les tool_calls en attente
                    run_id = event.get("run_id", "")
                    tool_call_id = ""
                    if hasattr(output, "tool_call_id"):
                        tool_call_id = output.tool_call_id
                    if tool_call_id in pending_tool_calls:
                        current_step.input = str(
                            pending_tool_calls[tool_call_id].get("args", "")
                        )[:500]

                    # Troncature de la sortie pour ne pas surcharger l'UI
                    content = output.content if hasattr(output, "content") else str(output)
                    current_step.output = str(content)[:1000]
                    await current_step.send()

                    # Reset du streaming pour la prochaine réponse de l'agent
                    if streaming_msg:
                        streaming_msg = None
                        is_streaming_text = False
                        final_response = ""

    except GraphRecursionError:
        # L'agent a atteint la limite de récursion (trop d'appels d'outils)
        if not final_response.strip():
            final_response = (
                "J'ai atteint la limite d'étapes pour cette requête. "
                "Voici ce que j'ai pu trouver jusqu'ici. "
                "N'hésite pas à reformuler ta demande si tu veux que je continue."
            )
    except Exception as e:
        # En cas d'erreur du graphe, on affiche un message d'erreur clair
        final_response = f"Désolé, une erreur s'est produite : {str(e)}"

    # --- Affichage de la réponse finale ---
    # Si le streaming a déjà affiché la réponse, on ne la réaffiche pas
    if streaming_msg and final_response.strip():
        # La réponse a déjà été streamée dans le chat — rien à faire
        pass
    elif final_response.strip():
        await cl.Message(content=final_response).send()
    else:
        # Si pas de réponse textuelle (ne devrait pas arriver), message par défaut
        await cl.Message(
            content="J'ai terminé l'exécution, mais je n'ai pas de réponse textuelle à afficher."
        ).send()


# =============================================================================
# ÉVÉNEMENT : FIN DU CHAT (optionnel)
# =============================================================================

@cl.on_chat_end
async def on_chat_end():
    """Appelé quand l'utilisateur ferme le chat ou se déconnecte.

    Sauvegarde la mémoire long-terme sur disque pour persister les
    apprentissages entre les sessions.
    """
    global _store

    if _store:
        from agent.memory import save_memory_store
        save_memory_store(_store)
