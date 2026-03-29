"""
ProjectX - Package Outils (Tools)
Ce package regroupe tous les outils que l'agent AI peut utiliser pour
interagir avec le monde extérieur : web, fichiers, système, email, code.

Chaque outil est un module indépendant utilisant le décorateur @tool de
LangChain, ce qui permet au LLM de les appeler automatiquement via le
mécanisme de tool-calling.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

# =============================================================================
# IMPORTS DES OUTILS
# On importe chaque outil individuellement pour les rendre accessibles
# via `from tools import ALL_TOOLS` dans le graphe de l'agent.
# =============================================================================

from tools.web_search import web_search
from tools.web_scraper import scrape_webpage
from tools.browser import (
    browser_navigate,
    browser_click,
    browser_fill,
    browser_screenshot,
    browser_get_content,
    browser_close,
)
from tools.file_manager import (
    read_file,
    write_file,
    list_directory,
    create_directory,
    delete_file,
    move_file,
)
from tools.system_cmd import run_command
from tools.email_tool import send_email
from tools.code_executor import execute_python
from tools.rag_tool import rag_index_documents, rag_search
from tools.memory_tool import save_memory, recall_memory

# =============================================================================
# LISTE COMPLÈTE DES OUTILS
# Cette liste est utilisée par le graphe LangGraph pour lier tous les outils
# au LLM via `bind_tools()`. L'ordre n'a pas d'importance.
# =============================================================================

ALL_TOOLS = [
    # --- Outils Web ---
    web_search,             # Recherche DuckDuckGo (sans API key)
    scrape_webpage,         # Extraction de texte depuis une URL
    browser_navigate,       # Naviguer vers une URL (Playwright)
    browser_click,          # Cliquer sur un élément dans le navigateur
    browser_fill,           # Remplir un champ de formulaire
    browser_screenshot,     # Prendre une capture d'écran
    browser_get_content,    # Récupérer le texte de la page courante
    browser_close,          # Fermer le navigateur

    # --- Outils Fichiers ---
    read_file,              # Lire le contenu d'un fichier
    write_file,             # Écrire du contenu dans un fichier
    list_directory,         # Lister le contenu d'un répertoire
    create_directory,       # Créer un répertoire
    delete_file,            # Supprimer un fichier ou dossier
    move_file,              # Déplacer/renommer un fichier

    # --- Outils Système ---
    run_command,            # Exécuter une commande système (cmd/powershell)
    send_email,             # Envoyer un email via SMTP

    # --- Outils Code ---
    execute_python,         # Exécuter du code Python dynamiquement

    # --- Outils RAG (Recherche dans les fichiers locaux) ---
    rag_index_documents,    # Indexer les fichiers d'un répertoire
    rag_search,             # Rechercher dans les documents indexés

    # --- Outils Mémoire Explicite ---
    save_memory,            # Sauvegarder un fait/information en mémoire permanente
    recall_memory,          # Retrouver des informations mémorisées
]
