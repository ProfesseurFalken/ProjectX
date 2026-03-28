"""
ProjectX - Outil de Gestion de Fichiers
Permet à l'agent de manipuler des fichiers et répertoires sur le système
local : lecture, écriture, listage, création, suppression et déplacement.

Utilise pathlib pour une gestion robuste et cross-platform des chemins.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import shutil
from pathlib import Path

from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Lit et retourne le contenu d'un fichier texte.

    Utilise cette fonction pour consulter le contenu de n'importe quel
    fichier texte sur le système (code source, configuration, logs, etc.).

    Args:
        file_path: Chemin absolu ou relatif vers le fichier à lire.
                   Exemples : "C:/Users/doc.txt", "./notes.md", "config.yaml".

    Returns:
        Le contenu complet du fichier, ou un message d'erreur si le fichier
        n'existe pas ou n'est pas lisible.
    """
    try:
        path = Path(file_path)

        # Vérification de l'existence du fichier avant lecture
        if not path.exists():
            return f"Erreur : le fichier '{file_path}' n'existe pas."

        if not path.is_file():
            return f"Erreur : '{file_path}' n'est pas un fichier (c'est peut-être un dossier)."

        # Lecture du contenu avec encodage UTF-8 (standard pour le texte)
        content = path.read_text(encoding="utf-8")

        return f"Contenu de '{file_path}' ({len(content)} caractères) :\n\n{content}"

    except UnicodeDecodeError:
        return f"Erreur : '{file_path}' n'est pas un fichier texte (binaire ?)."
    except PermissionError:
        return f"Erreur : permission refusée pour lire '{file_path}'."
    except Exception as e:
        return f"Erreur lors de la lecture de '{file_path}' : {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Écrit du contenu dans un fichier (crée le fichier s'il n'existe pas).

    Utilise cette fonction pour créer de nouveaux fichiers ou remplacer
    le contenu de fichiers existants. Les répertoires parents sont créés
    automatiquement si nécessaire.

    ATTENTION : cette fonction ÉCRASE le contenu existant du fichier.
    Pour ajouter du texte à la fin, lis d'abord le fichier puis réécris
    le contenu complet.

    Args:
        file_path: Chemin absolu ou relatif vers le fichier à écrire.
        content: Le contenu textuel à écrire dans le fichier.

    Returns:
        Confirmation de l'écriture avec le nombre de caractères écrits,
        ou message d'erreur.
    """
    try:
        path = Path(file_path)

        # Création automatique des répertoires parents s'ils n'existent pas
        # Exemple : write_file("dossier/sous-dossier/fichier.txt", "contenu")
        # crée automatiquement "dossier/" et "dossier/sous-dossier/"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Écriture du contenu avec encodage UTF-8
        path.write_text(content, encoding="utf-8")

        return f"Fichier '{file_path}' écrit avec succès ({len(content)} caractères)."

    except PermissionError:
        return f"Erreur : permission refusée pour écrire dans '{file_path}'."
    except Exception as e:
        return f"Erreur lors de l'écriture dans '{file_path}' : {str(e)}"


@tool
def list_directory(dir_path: str = ".") -> str:
    """Liste le contenu d'un répertoire (fichiers et sous-dossiers).

    Utilise cette fonction pour explorer l'arborescence de fichiers et
    voir ce qui se trouve dans un répertoire donné.

    Args:
        dir_path: Chemin du répertoire à lister. Par défaut "." (répertoire
                  courant). Exemples : "C:/Users/Bureau", "./src", "/tmp".

    Returns:
        Liste formatée des fichiers et dossiers avec leur type et taille,
        ou message d'erreur.
    """
    try:
        path = Path(dir_path)

        # Vérification de l'existence et du type
        if not path.exists():
            return f"Erreur : le répertoire '{dir_path}' n'existe pas."

        if not path.is_dir():
            return f"Erreur : '{dir_path}' n'est pas un répertoire."

        # Listage du contenu avec tri : dossiers d'abord, puis fichiers
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

        if not items:
            return f"Le répertoire '{dir_path}' est vide."

        # Formatage avec des indicateurs visuels pour chaque type
        formatted_items = []
        for item in items:
            if item.is_dir():
                # 📁 pour les dossiers
                formatted_items.append(f"  [DOSSIER] {item.name}/")
            else:
                # 📄 pour les fichiers, avec la taille en octets
                size = item.stat().st_size
                # Formatage intelligent de la taille
                if size < 1024:
                    size_str = f"{size} o"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} Ko"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} Mo"
                formatted_items.append(f"  [FICHIER] {item.name} ({size_str})")

        # Comptage pour le résumé
        nb_dirs = sum(1 for item in items if item.is_dir())
        nb_files = sum(1 for item in items if item.is_file())

        header = (
            f"Contenu de '{dir_path}' "
            f"({nb_dirs} dossier(s), {nb_files} fichier(s)) :\n"
        )
        return header + "\n".join(formatted_items)

    except PermissionError:
        return f"Erreur : permission refusée pour lister '{dir_path}'."
    except Exception as e:
        return f"Erreur lors du listage de '{dir_path}' : {str(e)}"


@tool
def create_directory(dir_path: str) -> str:
    """Crée un nouveau répertoire (avec les parents si nécessaire).

    Utilise cette fonction pour créer un ou plusieurs niveaux de dossiers.
    Si le dossier existe déjà, aucune erreur n'est levée.

    Args:
        dir_path: Chemin du répertoire à créer.
                  Exemples : "nouveau_dossier", "parent/enfant/petit-enfant".

    Returns:
        Confirmation de la création ou message d'erreur.
    """
    try:
        path = Path(dir_path)

        # parents=True : crée tous les dossiers intermédiaires
        # exist_ok=True : pas d'erreur si le dossier existe déjà
        path.mkdir(parents=True, exist_ok=True)

        return f"Répertoire '{dir_path}' créé avec succès."

    except PermissionError:
        return f"Erreur : permission refusée pour créer '{dir_path}'."
    except Exception as e:
        return f"Erreur lors de la création de '{dir_path}' : {str(e)}"


@tool
def delete_file(file_path: str) -> str:
    """Supprime un fichier ou un répertoire (et tout son contenu).

    ATTENTION : cette action est IRRÉVERSIBLE. Les fichiers supprimés
    ne vont PAS dans la corbeille.

    Args:
        file_path: Chemin du fichier ou répertoire à supprimer.

    Returns:
        Confirmation de la suppression ou message d'erreur.
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return f"Erreur : '{file_path}' n'existe pas."

        if path.is_file():
            # Suppression d'un fichier simple
            path.unlink()
            return f"Fichier '{file_path}' supprimé avec succès."
        elif path.is_dir():
            # Suppression récursive d'un dossier et tout son contenu
            # shutil.rmtree est nécessaire car Path.rmdir() ne supprime
            # que les dossiers vides
            shutil.rmtree(path)
            return f"Répertoire '{file_path}' et son contenu supprimés avec succès."

    except PermissionError:
        return f"Erreur : permission refusée pour supprimer '{file_path}'."
    except Exception as e:
        return f"Erreur lors de la suppression de '{file_path}' : {str(e)}"

    return f"Erreur : type de fichier non reconnu pour '{file_path}'."


@tool
def move_file(source: str, destination: str) -> str:
    """Déplace ou renomme un fichier ou répertoire.

    Utilise cette fonction pour déplacer un fichier vers un autre emplacement
    ou pour le renommer. Fonctionne aussi pour les répertoires.

    Args:
        source: Chemin actuel du fichier/répertoire à déplacer.
        destination: Nouveau chemin de destination.

    Returns:
        Confirmation du déplacement ou message d'erreur.
    """
    try:
        src = Path(source)
        dst = Path(destination)

        if not src.exists():
            return f"Erreur : la source '{source}' n'existe pas."

        # Création du répertoire parent de destination si nécessaire
        dst.parent.mkdir(parents=True, exist_ok=True)

        # shutil.move gère le déplacement cross-filesystem
        # (contrairement à Path.rename qui ne fonctionne que sur le même FS)
        shutil.move(str(src), str(dst))

        return f"'{source}' déplacé vers '{destination}' avec succès."

    except PermissionError:
        return f"Erreur : permission refusée pour déplacer '{source}'."
    except Exception as e:
        return f"Erreur lors du déplacement de '{source}' : {str(e)}"
