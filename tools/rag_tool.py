"""
ProjectX - RAG (Retrieval-Augmented Generation) sur les fichiers locaux
Permet à Joshua d'indexer et de rechercher dans les documents locaux
(texte, markdown, code, CSV, etc.) via des embeddings Ollama + ChromaDB.

L'indexation est déclenchée automatiquement au démarrage ou manuellement
via l'outil rag_index_documents. La recherche est disponible via l'outil
rag_search qui retourne les passages les plus pertinents.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import os
import hashlib
from pathlib import Path

from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

from config import (
    OLLAMA_BASE_URL,
    RAG_DOCUMENTS_DIR,
    RAG_FILE_EXTENSIONS,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
    RAG_CHROMA_DIR,
)

# Singleton ChromaDB client + collection
_chroma_collection = None


def _get_embeddings():
    """Retourne une instance d'embeddings Ollama nomic-embed-text."""
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_BASE_URL,
    )


def _get_collection():
    """Retourne la collection ChromaDB, la crée si nécessaire."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    import chromadb
    from chromadb.config import Settings

    # Création du client ChromaDB persistant sur disque
    client = chromadb.PersistentClient(
        path=RAG_CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # Récupération ou création de la collection
    _chroma_collection = client.get_or_create_collection(
        name="joshua_documents",
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


def _compute_file_hash(filepath: str) -> str:
    """Calcule un hash MD5 du contenu du fichier pour détecter les changements."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_file_content(filepath: str) -> str:
    """Lit le contenu textuel d'un fichier avec gestion des encodages."""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


@tool
def rag_index_documents(directory: str = "") -> str:
    """Indexe les fichiers d'un répertoire dans la base vectorielle pour la recherche RAG.

    Parcourt le répertoire spécifié (ou RAG_DOCUMENTS_DIR par défaut), lit
    chaque fichier supporté, le découpe en chunks, calcule les embeddings
    et les stocke dans ChromaDB. Les fichiers déjà indexés (même hash) sont
    ignorés pour éviter les doublons.

    Args:
        directory: Chemin du répertoire à indexer. Si vide, utilise
                   le répertoire par défaut configuré dans config.py.

    Returns:
        Un message résumant le nombre de fichiers indexés et les chunks créés.
    """
    target_dir = directory if directory else RAG_DOCUMENTS_DIR

    # Création du répertoire s'il n'existe pas
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    collection = _get_collection()
    embeddings = _get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    files_indexed = 0
    chunks_created = 0
    skipped = 0
    errors = []

    for root, _dirs, files in os.walk(target_dir):
        for filename in files:
            # Vérifier l'extension
            ext = Path(filename).suffix.lower()
            if ext not in RAG_FILE_EXTENSIONS:
                continue

            filepath = os.path.join(root, filename)
            file_hash = _compute_file_hash(filepath)

            # Vérifier si le fichier est déjà indexé avec le même hash
            existing = collection.get(
                where={"file_hash": file_hash},
                limit=1,
            )
            if existing and existing["ids"]:
                skipped += 1
                continue

            # Lire le contenu du fichier
            content = _read_file_content(filepath)
            if not content.strip():
                continue

            # Supprimer les anciens chunks de ce fichier s'ils existent
            try:
                old_chunks = collection.get(
                    where={"source": filepath},
                )
                if old_chunks and old_chunks["ids"]:
                    collection.delete(ids=old_chunks["ids"])
            except Exception:
                pass

            # Découper en chunks
            chunks = splitter.split_text(content)
            if not chunks:
                continue

            try:
                # Calculer les embeddings
                chunk_embeddings = embeddings.embed_documents(chunks)

                # Stocker dans ChromaDB
                ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
                metadatas = [
                    {
                        "source": filepath,
                        "filename": filename,
                        "file_hash": file_hash,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                    for i in range(len(chunks))
                ]

                collection.add(
                    ids=ids,
                    embeddings=chunk_embeddings,  # type: ignore[arg-type]
                    documents=chunks,
                    metadatas=metadatas,  # type: ignore[arg-type]
                )

                files_indexed += 1
                chunks_created += len(chunks)

            except Exception as e:
                errors.append(f"{filename}: {str(e)}")

    result = (
        f"Indexation terminée :\n"
        f"- Répertoire : {target_dir}\n"
        f"- Fichiers indexés : {files_indexed}\n"
        f"- Chunks créés : {chunks_created}\n"
        f"- Fichiers déjà à jour : {skipped}"
    )
    if errors:
        result += f"\n- Erreurs : {len(errors)}\n  " + "\n  ".join(errors[:5])

    return result


@tool
def rag_search(query: str) -> str:
    """Recherche dans les documents locaux indexés par similarité sémantique.

    Utilise les embeddings nomic-embed-text et ChromaDB pour trouver les
    passages de documents les plus pertinents par rapport à la requête.

    Utilise cet outil quand l'utilisateur pose une question sur ses fichiers
    locaux, ses documents, ses notes, ou tout contenu stocké sur son ordinateur.

    Args:
        query: La question ou le sujet à rechercher dans les documents locaux.

    Returns:
        Les passages les plus pertinents trouvés, avec le nom du fichier source
        et un score de similarité. Retourne un message si aucun document
        n'a été indexé.
    """
    collection = _get_collection()

    # Vérifier que des documents sont indexés
    if collection.count() == 0:
        return (
            "Aucun document n'est indexé dans la base RAG. "
            "Utilise rag_index_documents pour indexer un répertoire de fichiers."
        )

    embeddings = _get_embeddings()

    try:
        # Calcul de l'embedding de la requête
        query_embedding = embeddings.embed_query(query)

        # Recherche par similarité dans ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=RAG_TOP_K,
        )

        if not results["documents"] or not results["documents"][0]:
            return "Aucun résultat pertinent trouvé dans les documents locaux."

        # Formatage des résultats
        output_parts = [f"Résultats de recherche RAG pour : \"{query}\"\n"]

        documents = results["documents"][0]
        metadatas_list = results["metadatas"][0] if results["metadatas"] else []
        distances_list = results["distances"][0] if results["distances"] else []

        for i, doc in enumerate(documents):
            meta = metadatas_list[i] if i < len(metadatas_list) else {}
            dist = distances_list[i] if i < len(distances_list) else 0.0
            similarity = max(0, 1 - float(dist))
            source = meta.get("filename", "inconnu") if isinstance(meta, dict) else "inconnu"
            chunk_idx = int(meta.get("chunk_index", 0) or 0) if isinstance(meta, dict) else 0  # type: ignore[arg-type]
            total = meta.get("total_chunks", "?") if isinstance(meta, dict) else "?"

            output_parts.append(
                f"--- Résultat {i+1} (similarité: {similarity:.0%}) ---\n"
                f"Source : {source} (chunk {chunk_idx+1}/{total})\n"
                f"{doc}\n"
            )

        return "\n".join(output_parts)

    except Exception as e:
        return f"Erreur lors de la recherche RAG : {str(e)}"
