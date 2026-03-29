"""
ProjectX - Agents Spécialisés
Définit les spécialistes que l'orchestrateur peut invoquer.
Chaque spécialiste a son propre sous-ensemble d'outils et son prompt.

Architecture :
    Orchestrateur → [ResearchAgent | CoderAgent | SystemAgent | MemoryAgent | GeneralAgent]

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-29
"""

from tools import (
    web_search, scrape_webpage,
    browser_navigate, browser_click, browser_fill,
    browser_screenshot, browser_get_content, browser_close,
    read_file, write_file, list_directory, create_directory,
    delete_file, move_file,
    run_command, send_email,
    execute_python,
    rag_index_documents, rag_search,
    save_memory, recall_memory,
)

# =============================================================================
# DÉFINITION DES SPÉCIALISTES
# =============================================================================

SPECIALISTS = {
    "research": {
        "name": "ResearchAgent",
        "tools": [
            web_search, scrape_webpage,
            browser_navigate, browser_click, browser_fill,
            browser_screenshot, browser_get_content, browser_close,
            save_memory,
        ],
        "prompt": (
            "Tu es ResearchAgent, spécialiste de la recherche web. "
            "Tu utilises web_search pour trouver des informations, puis "
            "scrape_webpage pour lire les articles pertinents. "
            "Tu fais TOUJOURS une synthèse complète après avoir lu les sources. "
            "Ne retourne JAMAIS une simple liste de liens. "
            "Si tu trouves un fait important, appelle save_memory pour le retenir."
        ),
    },
    "coder": {
        "name": "CoderAgent",
        "tools": [
            execute_python, read_file, write_file,
            list_directory, create_directory,
            save_memory,
        ],
        "prompt": (
            "Tu es CoderAgent, spécialiste de la programmation. "
            "Tu écris, lis et exécutes du code Python. "
            "Tu gères les fichiers de code. "
            "Quand on te demande un calcul, utilise execute_python. "
            "Quand on te demande de créer un programme, utilise write_file "
            "puis execute_python pour le tester."
        ),
    },
    "system": {
        "name": "SystemAgent",
        "tools": [
            run_command, list_directory, create_directory,
            delete_file, move_file, read_file, write_file,
            send_email,
            save_memory,
        ],
        "prompt": (
            "Tu es SystemAgent, spécialiste des opérations système. "
            "Tu exécutes des commandes, gères les fichiers et dossiers, "
            "et envoies des emails. "
            "Sois prudent avec les commandes destructrices (suppression, etc.)."
        ),
    },
    "memory": {
        "name": "MemoryAgent",
        "tools": [
            save_memory, recall_memory,
            rag_index_documents, rag_search,
            read_file, list_directory,
        ],
        "prompt": (
            "Tu es MemoryAgent, spécialiste de la mémoire et du savoir. "
            "Tu sauvegardes et retrouves des informations avec save_memory "
            "et recall_memory. Tu indexes et cherches dans les documents "
            "locaux avec rag_index_documents et rag_search."
        ),
    },
    "general": {
        "name": "GeneralAgent",
        "tools": None,  # None = tous les outils (ALL_TOOLS)
        "prompt": (
            "Tu es Joshua, un agent AI autonome polyvalent. "
            "Utilise tes outils pour répondre de manière complète."
        ),
    },
}

# Mots-clés pour le routage heuristique rapide (fallback si LLM trop lent)
ROUTING_KEYWORDS = {
    "research": [
        "cherche", "recherche", "trouve", "google", "web", "internet",
        "article", "actualité", "news", "info sur", "qu'est-ce que",
        "c'est quoi", "qui est", "explique", "compare", "versus",
        "tendance", "récent", "nouveau", "dernière", "source",
        "site", "url", "lien", "page web", "naviguer",
    ],
    "coder": [
        "code", "python", "programme", "script", "fonction", "calcul",
        "algorithme", "bug", "debug", "erreur", "compile", "exécute",
        "variable", "boucle", "class", "import", "pip", "bibliothèque",
        "api", "json", "csv", "données", "analyse", "graphique", "plot",
    ],
    "system": [
        "commande", "terminal", "cmd", "powershell", "dossier", "fichier",
        "créer", "supprimer", "déplacer", "renommer", "copier", "installer",
        "email", "mail", "envoyer", "système", "disque", "processus",
        "service", "réseau", "ip", "ping", "port",
    ],
    "memory": [
        "souviens", "rappelle", "mémoire", "mémorise", "retiens",
        "n'oublie", "tu sais", "je t'ai dit", "mon nom", "je m'appelle",
        "indexe", "document", "rag", "connaissance", "apprends",
    ],
}
