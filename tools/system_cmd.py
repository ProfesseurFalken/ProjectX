"""
ProjectX - Outil d'Exécution de Commandes Système
Permet à l'agent d'exécuter des commandes système (PowerShell/cmd sur Windows,
bash/sh sur Linux/Mac) et d'en récupérer le résultat.

Inclut un timeout de sécurité pour éviter les commandes qui tournent
indéfiniment.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import subprocess
import platform

from langchain_core.tools import tool

from config import COMMAND_TIMEOUT


@tool
def run_command(command: str) -> str:
    """Exécute une commande système et retourne sa sortie.

    Utilise cette fonction pour exécuter des commandes dans le terminal
    du système d'exploitation : lister des processus, gérer des services,
    installer des paquets, manipuler Git, etc.

    Sur Windows, la commande est exécutée via PowerShell.
    Sur Linux/Mac, elle est exécutée via /bin/sh.

    Args:
        command: La commande à exécuter.
                 Exemples : "dir", "git status", "python --version",
                 "ipconfig", "ping google.com -n 4".

    Returns:
        La sortie standard (stdout) et la sortie d'erreur (stderr) de la
        commande, avec le code de retour. En cas de timeout, retourne un
        message d'erreur.
    """
    try:
        # Détection du système d'exploitation pour choisir le shell approprié
        # Windows utilise PowerShell pour une meilleure compatibilité
        # Linux/Mac utilisent le shell par défaut (/bin/sh)
        is_windows = platform.system() == "Windows"

        # Exécution de la commande via subprocess
        # - shell=True : permet d'utiliser la syntaxe native du shell
        #   (pipes, redirections, variables d'environnement, etc.)
        # - capture_output=True : capture stdout et stderr
        # - text=True : décode la sortie en texte (pas en bytes)
        # - timeout : interrompt la commande après N secondes
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            # Sur Windows, on encode en cp1252/utf-8 pour les caractères spéciaux
            encoding="utf-8",
            errors="replace",  # Remplace les caractères non décodables par '?'
        )

        # Construction de la réponse avec stdout, stderr et code de retour
        output_parts = []

        # Stdout : sortie normale de la commande
        if result.stdout.strip():
            output_parts.append(f"Sortie :\n{result.stdout.strip()}")

        # Stderr : messages d'erreur ou d'information
        # Note : certains programmes envoient des informations normales
        # sur stderr (ex: git, curl), donc ce n'est pas forcément une erreur
        if result.stderr.strip():
            output_parts.append(f"Erreurs/Avertissements :\n{result.stderr.strip()}")

        # Code de retour : 0 = succès, autre = erreur
        output_parts.append(f"Code de retour : {result.returncode}")

        # Si aucune sortie du tout, on le signale
        if not result.stdout.strip() and not result.stderr.strip():
            output_parts.insert(0, "La commande s'est exécutée sans produire de sortie.")

        return f"Commande : {command}\n\n" + "\n\n".join(output_parts)

    except subprocess.TimeoutExpired:
        # La commande a dépassé le timeout — on l'interrompt
        return (
            f"Erreur : la commande '{command}' a dépassé le timeout "
            f"de {COMMAND_TIMEOUT} secondes et a été interrompue."
        )
    except Exception as e:
        return f"Erreur lors de l'exécution de '{command}' : {str(e)}"
