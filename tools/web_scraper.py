"""
ProjectX - Outil de Scraping Web
Permet à l'agent d'extraire le contenu textuel d'une page web donnée.
Utilise requests pour récupérer le HTML et BeautifulSoup pour l'analyser.
Inclut un système de retry automatique avec rotation de User-Agent en cas
d'échec (timeout, erreur HTTP, connexion refusée).

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import time
import random

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from config import SCRAPER_MAX_LENGTH, HTTP_TIMEOUT


# Liste de User-Agents réalistes pour la rotation
# En cas d'échec, on réessaye avec un User-Agent différent
# pour contourner les blocages basiques par user-agent
_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) "
        "Gecko/20100101 Firefox/122.0"
    ),
]

# Nombre maximum de tentatives avant d'abandonner
_MAX_RETRIES = 3

# Délai entre les tentatives (en secondes) — croissant
_RETRY_DELAYS = [0, 2, 5]


def _fetch_with_retry(url: str) -> requests.Response:
    """Effectue une requête GET avec retry automatique et rotation de User-Agent.

    En cas d'échec (timeout, erreur HTTP 5xx, connexion refusée), la fonction
    réessaye jusqu'à _MAX_RETRIES fois avec un User-Agent différent et un
    délai croissant entre les tentatives.

    Les erreurs HTTP 4xx (sauf 429 Too Many Requests) ne sont PAS réessayées
    car elles indiquent un problème côté client (URL invalide, page supprimée).

    Args:
        url: L'URL à récupérer.

    Returns:
        L'objet Response en cas de succès.

    Raises:
        requests.exceptions.RequestException: Si toutes les tentatives échouent.
    """
    last_error: requests.exceptions.RequestException | None = None

    for attempt in range(_MAX_RETRIES):
        # Délai avant la tentative (0 pour la première)
        if attempt > 0:
            time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

        # Rotation de User-Agent à chaque tentative
        headers = {
            "User-Agent": _USER_AGENTS[attempt % len(_USER_AGENTS)],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        try:
            response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)

            # Succès (2xx/3xx) → retourner directement
            if response.status_code < 400:
                return response

            # Erreur 429 (rate limit) → réessayer après un délai plus long
            if response.status_code == 429:
                last_error = requests.exceptions.HTTPError(
                    f"429 Too Many Requests", response=response
                )
                time.sleep(5)  # Attente supplémentaire pour rate limit
                continue

            # Erreur 5xx (serveur) → réessayer
            if response.status_code >= 500:
                last_error = requests.exceptions.HTTPError(
                    f"{response.status_code}", response=response
                )
                continue

            # Erreur 4xx (client, sauf 429) → ne pas réessayer
            response.raise_for_status()

        except requests.exceptions.Timeout as e:
            last_error = e
            continue  # Timeout → réessayer
        except requests.exceptions.ConnectionError as e:
            last_error = e
            continue  # Connexion refusée → réessayer
        except requests.exceptions.HTTPError:
            raise  # 4xx → ne pas réessayer, remonter directement

    # Toutes les tentatives ont échoué → remonter la dernière erreur
    if last_error is not None:
        raise last_error
    raise requests.exceptions.ConnectionError(f"Échec après {_MAX_RETRIES} tentatives pour {url}")


@tool
def scrape_webpage(url: str) -> str:
    """Extrait le contenu textuel principal d'une page web.

    Utilise cette fonction quand tu as besoin de lire le contenu détaillé
    d'une page web spécifique (article, documentation, page produit, etc.).
    Le HTML est nettoyé et seul le texte pertinent est retourné.
    En cas d'échec, la requête est automatiquement réessayée jusqu'à 3 fois
    avec rotation de User-Agent.

    Args:
        url: L'URL complète de la page web à scraper
             (ex: "https://example.com/article").

    Returns:
        Le texte principal de la page, nettoyé du HTML, des scripts et
        des styles. Le texte est tronqué à SCRAPER_MAX_LENGTH caractères
        si la page est très longue. En cas d'erreur, retourne un message
        d'erreur descriptif.
    """
    try:
        # --- Étape 1 : Récupération du HTML avec retry automatique ---
        response = _fetch_with_retry(url)

        # --- Étape 2 : Parsing et nettoyage du HTML ---
        # BeautifulSoup analyse le HTML et nous permet d'en extraire le texte
        soup = BeautifulSoup(response.text, "html.parser")

        # Suppression des éléments non pertinents qui polluent le texte :
        # - <script>  : code JavaScript
        # - <style>   : feuilles de style CSS
        # - <nav>     : barres de navigation
        # - <footer>  : pieds de page
        # - <header>  : en-têtes de site (logo, menu)
        # - <aside>   : barres latérales (publicités, widgets)
        for element in soup.find_all(
            ["script", "style", "nav", "footer", "header", "aside"]
        ):
            element.decompose()  # Supprime l'élément et son contenu du DOM

        # --- Étape 3 : Extraction du texte ---
        # get_text() extrait tout le texte visible de la page
        # separator="\n" : un saut de ligne entre chaque bloc
        # strip=True : supprime les espaces en début/fin de chaque ligne
        text = soup.get_text(separator="\n", strip=True)

        # Nettoyage : suppression des lignes vides multiples
        # On ne garde que les lignes non vides pour un texte propre
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        # --- Étape 4 : Troncature si nécessaire ---
        # Les pages web peuvent être très longues ; on tronque pour ne pas
        # saturer la fenêtre de contexte du LLM Ollama
        if len(clean_text) > SCRAPER_MAX_LENGTH:
            clean_text = clean_text[:SCRAPER_MAX_LENGTH] + "\n\n[... contenu tronqué]"

        return f"Contenu de {url} :\n\n{clean_text}"

    except requests.exceptions.Timeout:
        return f"Erreur : timeout après {HTTP_TIMEOUT}s en accédant à {url}"
    except requests.exceptions.HTTPError as e:
        return f"Erreur HTTP {e.response.status_code} en accédant à {url}"
    except requests.exceptions.ConnectionError:
        return f"Erreur de connexion : impossible d'accéder à {url}"
    except Exception as e:
        return f"Erreur lors du scraping de {url} : {str(e)}"
