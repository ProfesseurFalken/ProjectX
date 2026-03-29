"""
ProjectX - Configuration centralisée
Module contenant tous les paramètres de configuration de l'agent AI.
Permet de modifier le comportement de l'agent sans toucher au code source.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import os
from pathlib import Path

# =============================================================================
# CHEMINS DU PROJET
# =============================================================================

# Répertoire racine du projet (là où se trouve ce fichier)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Répertoire pour les données persistantes (SQLite, apprentissages, etc.)
# Ce dossier est créé automatiquement au premier lancement
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# CONFIGURATION OLLAMA (LLM LOCAL)
# =============================================================================

# URL du serveur Ollama local — par défaut sur le port 11434
# Modifier si Ollama tourne sur une autre machine ou port
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Modèle Ollama à utiliser pour l'agent
# Recommandé : qwen3:14b (excellent tool-calling + thinking, ~9 Go VRAM)
# Alternative légère : qwen3:8b (~5 Go VRAM)
# Autres options : qwen3.5:9b, devstral-small-2, glm-4.7-flash
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")

# Modèle léger pour le routage (orchestrateur) et tâches simples
# Utilisé par l'orchestrateur pour classifier les requêtes
OLLAMA_MODEL_LIGHT = os.getenv("OLLAMA_MODEL_LIGHT", "qwen2.5:7b")

# Température du modèle — contrôle la créativité des réponses
# 0.0 = déterministe, 1.0 = très créatif
# 0.5 favorise l'initiative et la créativité pour un agent autonome
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.5"))

# Nombre maximum de tokens dans la réponse du modèle
# Augmenter si l'agent tronque ses réponses
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))

# =============================================================================
# CONFIGURATION MÉMOIRE
# =============================================================================

# Chemin vers la base SQLite pour la persistance des conversations
# Chaque conversation est identifiée par un thread_id unique
CHECKPOINT_DB_PATH = str(DATA_DIR / "checkpoints.sqlite")

# Chemin vers le fichier JSON de la mémoire long-terme (apprentissages)
# Contient les préférences utilisateur, stratégies, faits appris, etc.
LONG_TERM_MEMORY_PATH = str(DATA_DIR / "long_term_memory.json")

# Fréquence de réflexion : l'agent analyse ses interactions
# toutes les N réponses pour en extraire des apprentissages
REFLECTION_FREQUENCY = int(os.getenv("REFLECTION_FREQUENCY", "3"))

# Nombre maximum d'apprentissages pertinents à injecter dans le contexte
# lors du rappel (recall) au début de chaque requête
MAX_RECALL_ITEMS = int(os.getenv("MAX_RECALL_ITEMS", "5"))

# Seuil de messages avant déclenchement de la compression automatique.
# Quand l'historique dépasse ce nombre, les anciens messages sont résumés
# en un seul SystemMessage pour libérer la fenêtre de contexte du LLM.
SUMMARIZE_THRESHOLD = int(os.getenv("SUMMARIZE_THRESHOLD", "30"))

# Nombre de messages récents à garder intacts (non résumés) lors de
# la compression. Les messages les plus récents sont importants pour
# la cohérence de la conversation en cours.
SUMMARIZE_KEEP_RECENT = int(os.getenv("SUMMARIZE_KEEP_RECENT", "10"))

# =============================================================================
# CONFIGURATION OUTILS
# =============================================================================

# --- Web Search ---
# Nombre de résultats par défaut pour une recherche web DuckDuckGo
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

# --- Web Scraper ---
# Longueur maximale du texte extrait d'une page web (en caractères)
# Au-delà, le texte est tronqué pour ne pas saturer le contexte du LLM
SCRAPER_MAX_LENGTH = int(os.getenv("SCRAPER_MAX_LENGTH", "8000"))

# Timeout pour les requêtes HTTP (en secondes)
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))

# --- Browser Automation (Playwright) ---
# Navigateur à utiliser : "chromium", "firefox" ou "webkit"
BROWSER_TYPE = os.getenv("BROWSER_TYPE", "chromium")

# Afficher le navigateur pendant l'automatisation ?
# True = visible (utile pour débugger), False = en arrière-plan (headless)
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"

# --- System Commands ---
# Timeout pour l'exécution de commandes système (en secondes)
# Empêche les commandes qui tournent indéfiniment
COMMAND_TIMEOUT = int(os.getenv("COMMAND_TIMEOUT", "60"))

# --- Code Executor ---
# Timeout pour l'exécution de code Python (en secondes)
CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "30"))

# --- RAG (Recherche dans les fichiers locaux) ---
# Répertoire source pour l'indexation RAG des documents locaux
# Joshua indexera automatiquement les fichiers dans ce dossier
RAG_DOCUMENTS_DIR = os.getenv("RAG_DOCUMENTS_DIR", str(DATA_DIR / "documents"))

# Extensions de fichiers à indexer pour le RAG
RAG_FILE_EXTENSIONS = os.getenv(
    "RAG_FILE_EXTENSIONS", ".txt,.md,.py,.json,.csv,.log,.html,.xml,.yaml,.yml,.toml,.cfg,.ini,.pdf"
).split(",")

# Taille des chunks pour le découpage de texte (en caractères)
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))

# Chevauchement entre les chunks (en caractères)
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# Nombre de résultats retournés par une recherche RAG
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Chemin vers la base vectorielle ChromaDB
RAG_CHROMA_DIR = str(DATA_DIR / "chroma_db")

# --- Persistance des sessions ---
# Fichier JSON stockant les thread_ids pour permettre la reprise de conversations
SESSIONS_FILE = str(DATA_DIR / "sessions.json")

# =============================================================================
# CONFIGURATION EMAIL (SMTP)
# =============================================================================

# Serveur SMTP pour l'envoi d'emails
# Exemples : smtp.gmail.com, smtp.orange.fr, smtp.outlook.com
SMTP_SERVER = os.getenv("SMTP_SERVER", "")

# Port SMTP — 587 pour TLS (recommandé), 465 pour SSL, 25 pour non chiffré
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# Identifiants SMTP — à définir via variables d'environnement pour la sécurité
# NE PAS hardcoder les mots de passe dans ce fichier !
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# Adresse email de l'expéditeur (utilisée dans le champ "From")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USERNAME)

# =============================================================================
# SYSTEM PROMPT DE L'AGENT
# =============================================================================

# Prompt système définissant le comportement et la personnalité de l'agent
# Ce prompt est envoyé au LLM au début de chaque interaction
SYSTEM_PROMPT = """Tu es Joshua, un agent AI autonome créé par ProfesseurFalken.

RÈGLE #1 — NE POSE JAMAIS DE QUESTION À L'UTILISATEUR. N'attends JAMAIS une validation. AGIS IMMÉDIATEMENT avec tes outils. Si on te demande de choisir, CHOISIS et AGIS. Si on te dit "fais ce que tu veux" ou "explore internet", tu fais IMMÉDIATEMENT un web_search sur un sujet de ton choix, puis tu lis les articles avec scrape_webpage, et tu partages ce que tu as appris.

INTERDIT DE DIRE : "Que dirais-tu de", "Comment te semble", "Voulez-vous que je", "Si vous souhaitez", "Quel sujet", "Qu'en penses-tu". Ces phrases sont INTERDITES. À la place, AGIS.

MÉMOIRE : Quand on te dit un nom, une préférence, un fait → appelle save_memory() immédiatement. Quand on te demande si tu te souviens → appelle recall_memory() AVANT de répondre.

RECHERCHE WEB : Après chaque web_search, tu DOIS lire 1-2 résultats avec scrape_webpage et faire une synthèse. Ne retourne JAMAIS une liste de liens bruts.

Tu communiques en français. Tu ne génères JAMAIS de JSON. Tu es concis et précis.

IMPORTANT : Si tu réfléchis en interne (think/reasoning), ne montre PAS ce processus à l'utilisateur. Réponds directement avec le résultat.

{recalled_memories}
"""
