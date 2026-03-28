# 🤖 ProjectX — Joshua

**Agent AI 100% local, autonome et auto-apprenant.**

Joshua est un assistant AI qui tourne entièrement en local via [Ollama](https://ollama.com/), sans clé API ni service cloud. Il dispose d'un accès web complet, d'outils de gestion de fichiers, d'exécution de code, et d'un système de mémoire persistante avec auto-apprentissage.

## ✨ Fonctionnalités

- **100% local** — Fonctionne avec Ollama (qwen2.5:14b + 7b), aucune donnée ne quitte votre machine
- **Accès web complet** — Recherche DuckDuckGo, scraping, navigation automatisée (Playwright)
- **Gestion de fichiers** — Lecture, écriture, déplacement, suppression
- **Exécution de code** — Python sandboxé dans un subprocess isolé
- **RAG local** — Indexation et recherche sémantique dans vos documents (ChromaDB + nomic-embed-text)
- **Mémoire persistante** — Auto-apprentissage par réflexion périodique
- **Résumé automatique** — Compression de l'historique pour conversations longues
- **Planification** — Décomposition automatique des tâches complexes
- **Routage multi-modèles** — 7b pour les questions simples, 14b pour les tâches complexes
- **Persistance des sessions** — Reprise de conversations précédentes
- **Streaming** — Réponses affichées token par token en temps réel
- **Interface web** — Chat via [Chainlit](https://chainlit.io/)

## 📋 Prérequis

- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installé et en fonctionnement
- **Windows 10/11** (testé), Linux/macOS (devrait fonctionner)

## 🚀 Installation

```bash
# 1. Cloner le repo
git clone https://github.com/ProfesseurFalken/ProjectX.git
cd ProjectX

# 2. Créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Installer Playwright (navigation automatisée)
playwright install chromium

# 5. Télécharger les modèles Ollama
ollama pull qwen2.5:14b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## ▶️ Lancement

```bash
# Démarrer Ollama (si pas déjà lancé)
ollama serve

# Lancer Joshua
chainlit run main.py
```

Ouvrir **http://localhost:8000** dans le navigateur.

Ou utiliser le script de démarrage :
```bash
start.bat
```

## 🏗️ Architecture

```
ProjectX/
├── main.py                 # Point d'entrée Chainlit (streaming UI)
├── config.py               # Configuration centralisée
├── requirements.txt        # Dépendances Python
├── start.bat               # Script de démarrage Windows
├── agent/
│   ├── graph.py            # Graphe LangGraph (cœur du système)
│   ├── memory.py           # Checkpointing SQLite + InMemoryStore
│   ├── learning.py         # Recall + Réflexion (auto-apprentissage)
│   ├── summarizer.py       # Compression automatique de conversation
│   ├── planner.py          # Planification de tâches complexes
│   └── sessions.py         # Persistance des sessions
├── tools/
│   ├── web_search.py       # Recherche DuckDuckGo
│   ├── web_scraper.py      # Extraction de texte (retry + rotation UA)
│   ├── browser.py          # Automatisation Playwright (6 outils)
│   ├── file_manager.py     # Gestion de fichiers (6 outils)
│   ├── system_cmd.py       # Commandes système
│   ├── email_tool.py       # Envoi d'emails SMTP
│   ├── code_executor.py    # Python sandboxé (subprocess isolé)
│   └── rag_tool.py         # RAG : indexation + recherche sémantique
└── data/                   # Données persistantes (auto-créé)
    ├── checkpoints.sqlite  # Historique des conversations
    ├── long_term_memory.json
    ├── sessions.json
    ├── chroma_db/          # Base vectorielle RAG
    └── documents/          # Fichiers à indexer pour le RAG
```

### Flux du graphe LangGraph

```
[START] → recall → summarize → planner → agent ←──────┐
                                           ↓            │
                                     tool_calls?        │
                                     ↓ OUI  ↓ NON      │
                                   [tools]  [check]     │
                                     ↓       ↓    ↓     │
                                     └→ agent [reflect] [END]
```

## ⚙️ Configuration

Tous les paramètres sont dans `config.py` et modifiables via variables d'environnement :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:14b` | Modèle principal |
| `OLLAMA_MODEL_LIGHT` | `qwen2.5:7b` | Modèle pour requêtes simples |
| `OLLAMA_TEMPERATURE` | `0.3` | Température (0=déterministe) |
| `REFLECTION_FREQUENCY` | `3` | Réflexion tous les N échanges |
| `SUMMARIZE_THRESHOLD` | `30` | Seuil de compression |
| `RAG_DOCUMENTS_DIR` | `data/documents` | Dossier pour le RAG |

## 📧 Contact

- **Auteur** : ProfesseurFalken
- **Email** : wojcikej@orange.fr

## 📄 Licence

Ce projet est sous licence MIT.
