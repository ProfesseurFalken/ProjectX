"""
ProjectX - Outil de Recherche Web (DuckDuckGo)
Permet à l'agent de rechercher des informations sur le web en utilisant
DuckDuckGo comme moteur de recherche. Aucune clé API requise.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

from langchain_core.tools import tool
from ddgs import DDGS

from config import SEARCH_MAX_RESULTS


@tool
def web_search(query: str, max_results: int = SEARCH_MAX_RESULTS) -> str:
    """Recherche des informations sur le web via DuckDuckGo.

    Utilise cette fonction quand tu as besoin de trouver des informations
    actuelles sur internet : actualités, prix, définitions, tutoriels, etc.

    Args:
        query: La requête de recherche (ex: "prix bitcoin aujourd'hui",
               "dernières nouvelles intelligence artificielle").
        max_results: Nombre maximum de résultats à retourner (défaut: 5).

    Returns:
        Une chaîne formatée contenant les résultats de recherche avec
        le titre, un extrait et l'URL de chaque résultat.
        En cas d'erreur, retourne un message d'erreur descriptif.
    """
    try:
        # Création d'une instance DuckDuckGo Search
        # Pas besoin de clé API — DuckDuckGo est gratuit et respecte la vie privée
        ddgs = DDGS()

        # Exécution de la recherche textuelle
        # La méthode text() retourne une liste de dictionnaires avec :
        # - "title" : titre de la page
        # - "body"  : extrait/snippet du contenu
        # - "href"  : URL de la page
        results = list(ddgs.text(query, max_results=max_results))

        # Si aucun résultat trouvé, on informe l'agent
        if not results:
            return f"Aucun résultat trouvé pour la recherche : '{query}'"

        # Formatage des résultats en texte lisible pour le LLM
        # Chaque résultat est numéroté et contient titre, extrait et URL
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "Sans titre")
            snippet = result.get("body", "Pas de description")
            url = result.get("href", "URL inconnue")
            formatted_results.append(
                f"[{i}] {title}\n"
                f"    {snippet}\n"
                f"    URL: {url}"
            )

        return f"Résultats de recherche pour '{query}' :\n\n" + "\n\n".join(
            formatted_results
        )

    except Exception as e:
        # En cas d'erreur (réseau, rate limiting, etc.), on retourne
        # un message explicite pour que l'agent puisse réagir
        return f"Erreur lors de la recherche web : {str(e)}"
