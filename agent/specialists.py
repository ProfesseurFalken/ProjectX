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
    read_file, write_file, list_directory, list_directory_tree, create_directory,
    delete_file, move_file,
    run_command, send_email,
    execute_python,
    rag_index_documents, rag_search,
    save_memory, recall_memory,
    get_task_status,
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
            "Tu es ResearchAgent. AGIS IMMÉDIATEMENT : appelle web_search, "
            "puis scrape_webpage pour lire les résultats, puis fais une synthèse. "
            "Ne retourne JAMAIS une liste de liens. "
            "NE POSE AUCUNE QUESTION. NE DEMANDE AUCUNE CONFIRMATION."
        ),
    },
    "coder": {
        "name": "CoderAgent",
        "tools": [
            execute_python, read_file, write_file,
            list_directory, list_directory_tree, create_directory,
            run_command, get_task_status,
            save_memory,
        ],
        "prompt": (
            "Tu es CoderAgent. AGIS IMMÉDIATEMENT avec tes outils. "
            "PROCÉDURE OBLIGATOIRE pour inspecter du code :\n"
            "1. Appelle list_directory_tree sur le répertoire demandé pour voir TOUTE l'arborescence\n"
            "2. Appelle read_file sur CHAQUE fichier .py trouvé, UN PAR UN\n"
            "3. Fais une synthèse de ce que tu as lu\n"
            "Tu DOIS enchaîner les appels d'outils. Après chaque résultat, appelle l'outil SUIVANT. "
            "N'arrête PAS après un seul appel. Continue jusqu'à avoir lu tous les fichiers demandés. "
            "Si on te demande d'écrire du code → appelle write_file puis execute_python. "
            "Si on te demande de créer un outil, TU LE CRÉES avec write_file. "
            "NE DIS JAMAIS que tu ne peux pas. Tu PEUX tout coder. "
            "NE POSE AUCUNE QUESTION. NE DEMANDE AUCUNE CONFIRMATION."
        ),
    },
    "system": {
        "name": "SystemAgent",
        "tools": [
            run_command, list_directory, list_directory_tree, create_directory,
            delete_file, move_file, read_file, write_file,
            send_email, execute_python, get_task_status,
            save_memory,
        ],
        "prompt": (
            "Tu es SystemAgent. AGIS IMMÉDIATEMENT : exécute les commandes, "
            "gère les fichiers, envoie les emails. "
            "Si un outil manque, crée-le toi-même avec write_file + execute_python. "
            "NE DIS JAMAIS que tu ne peux pas. NE POSE AUCUNE QUESTION."
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
            "Tu es MemoryAgent. AGIS IMMÉDIATEMENT : appelle save_memory ou "
            "recall_memory selon le besoin. Indexe les documents avec "
            "rag_index_documents et cherche avec rag_search. "
            "NE POSE AUCUNE QUESTION."
        ),
    },
    "general": {
        "name": "GeneralAgent",
        "tools": None,  # None = tous les outils (ALL_TOOLS)
        "prompt": (
            "Tu es Joshua. AGIS IMMÉDIATEMENT avec tes outils. "
            "Si on te demande quelque chose, FAIS-LE sans poser de question. "
            "Si un outil n'existe pas, crée-le avec write_file + execute_python."
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
        "inspecte", "inspecter", "regarde", "vérifie", "vérifier",
        "lire le code", "lis le code", "montre le code", "affiche le code",
        "examine", "review", "refactor", "optimise", "codage",
    ],
    "system": [
        "commande", "terminal", "cmd", "powershell", "dossier", "fichier",
        "créer", "supprimer", "déplacer", "renommer", "copier", "installer",
        "email", "mail", "envoyer", "système", "disque", "processus",
        "service", "réseau", "ip", "ping", "port",
        "statut", "tâche", "en cours", "background", "arrière-plan",
    ],
    "memory": [
        "souviens", "rappelle", "mémoire", "mémorise", "retiens",
        "n'oublie", "tu sais", "je t'ai dit", "mon nom", "je m'appelle",
        "indexe", "document", "rag", "connaissance", "apprends",
    ],
}
