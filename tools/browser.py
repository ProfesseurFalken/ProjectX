"""
ProjectX - Outil d'Automatisation Navigateur (Playwright)
Permet à l'agent de contrôler un navigateur web complet : naviguer vers
des URLs, cliquer sur des éléments, remplir des formulaires, prendre des
captures d'écran, et extraire le contenu des pages.

Utilise Playwright en mode synchrone pour simplifier l'intégration avec
LangChain. Le navigateur est lancé en arrière-plan (headless) par défaut.

Prérequis : `playwright install chromium` doit être exécuté une fois.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import base64
from typing import Optional

from langchain_core.tools import tool
from playwright.sync_api import sync_playwright, Browser, Page

from config import BROWSER_TYPE, BROWSER_HEADLESS, DATA_DIR

# =============================================================================
# GESTION DU NAVIGATEUR SINGLETON
# Le navigateur est partagé entre tous les outils browser_* pour maintenir
# le contexte de navigation (cookies, session, historique) pendant la durée
# de vie de l'agent. Il est créé au premier appel et réutilisé ensuite.
# =============================================================================

# Variables globales pour le navigateur persistant
# On utilise un pattern singleton pour éviter de relancer le navigateur
# à chaque appel d'outil (coûteux en temps et mémoire)
_playwright_instance = None  # Instance Playwright (gestionnaire)
_browser: Optional[Browser] = None  # Instance du navigateur (Chromium/Firefox)
_page: Optional[Page] = None  # Page active (onglet courant)


def _get_page() -> Page:
    """Retourne la page active du navigateur, en le créant si nécessaire.

    Cette fonction gère le cycle de vie du navigateur singleton :
    - Premier appel : lance Playwright + navigateur + crée une page
    - Appels suivants : retourne la page existante

    Returns:
        L'objet Page Playwright sur lequel effectuer les actions.
    """
    global _playwright_instance, _browser, _page

    # Si le navigateur n'est pas encore lancé, on l'initialise
    if _page is None or _page.is_closed():
        # Démarrage de Playwright (gestionnaire de navigateurs)
        _playwright_instance = sync_playwright().start()

        # Sélection du type de navigateur selon la configuration
        # Playwright supporte : chromium, firefox, webkit
        browser_type = getattr(_playwright_instance, BROWSER_TYPE)

        # Lancement du navigateur
        # headless=True : pas de fenêtre visible (mode serveur)
        # headless=False : fenêtre visible (utile pour le debug)
        _browser = browser_type.launch(headless=BROWSER_HEADLESS)

        # Création d'une nouvelle page (onglet)
        if _browser is not None:
            _page = _browser.new_page()

    # Vérification que la page a bien été créée
    # (impossible en théorie, mais protège contre les cas inattendus)
    if _page is None:
        raise RuntimeError(
            "Impossible de créer une page navigateur. "
            "Vérifiez que Playwright est installé : playwright install chromium"
        )

    return _page


@tool
def browser_navigate(url: str) -> str:
    """Navigue vers une URL dans le navigateur automatisé.

    Utilise cette fonction pour ouvrir une page web dans le navigateur
    contrôlé par Playwright. Le navigateur maintient les cookies et la
    session entre les navigations.

    Args:
        url: L'URL complète vers laquelle naviguer
             (ex: "https://www.google.com").

    Returns:
        Confirmation avec le titre de la page chargée, ou message d'erreur.
    """
    try:
        page = _get_page()

        # Navigation avec attente du chargement complet de la page
        # wait_until="domcontentloaded" : attend que le DOM soit prêt
        # (plus rapide que "load" qui attend aussi images/CSS)
        page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Retour du titre de la page comme confirmation
        title = page.title()
        return f"Navigation réussie vers {url}\nTitre de la page : {title}"

    except Exception as e:
        return f"Erreur de navigation vers {url} : {str(e)}"


@tool
def browser_click(selector: str) -> str:
    """Clique sur un élément de la page web actuelle.

    Utilise cette fonction pour interagir avec des boutons, liens, ou
    tout élément cliquable sur la page actuellement chargée dans le
    navigateur.

    Args:
        selector: Sélecteur CSS ou texte de l'élément à cliquer.
                  Exemples : "#submit-btn", ".menu-item", "text=Se connecter",
                  "button:has-text('Valider')".

    Returns:
        Confirmation du clic ou message d'erreur si l'élément n'est pas trouvé.
    """
    try:
        page = _get_page()

        # Clic sur l'élément avec attente automatique qu'il soit visible
        # Playwright attend automatiquement que l'élément soit :
        # - Visible dans le viewport
        # - Stable (pas en cours d'animation)
        # - Activé (pas disabled)
        page.click(selector, timeout=10000)

        return f"Clic effectué sur l'élément : {selector}"

    except Exception as e:
        return f"Erreur lors du clic sur '{selector}' : {str(e)}"


@tool
def browser_fill(selector: str, value: str) -> str:
    """Remplit un champ de formulaire sur la page web actuelle.

    Utilise cette fonction pour saisir du texte dans des champs input,
    textarea, ou tout élément de formulaire éditable.

    Args:
        selector: Sélecteur CSS du champ à remplir.
                  Exemples : "#email", "input[name='username']",
                  "[placeholder='Rechercher']".
        value: Le texte à saisir dans le champ.

    Returns:
        Confirmation de la saisie ou message d'erreur.
    """
    try:
        page = _get_page()

        # fill() efface le contenu existant puis tape le nouveau texte
        # C'est plus fiable que type() qui ajoute au contenu existant
        page.fill(selector, value, timeout=10000)

        return f"Champ '{selector}' rempli avec : {value}"

    except Exception as e:
        return f"Erreur lors du remplissage de '{selector}' : {str(e)}"


@tool
def browser_screenshot() -> str:
    """Prend une capture d'écran de la page web actuelle.

    Utilise cette fonction quand tu as besoin de voir visuellement ce qui
    est affiché dans le navigateur. La capture est sauvegardée dans le
    dossier data/ du projet.

    Returns:
        Le chemin vers le fichier de capture d'écran, ou message d'erreur.
    """
    try:
        page = _get_page()

        # Chemin de sauvegarde dans le dossier data/
        screenshot_path = str(DATA_DIR / "screenshot.png")

        # Capture d'écran de la page entière (pas seulement le viewport)
        # full_page=True capture toute la page, même ce qui n'est pas visible
        page.screenshot(path=screenshot_path, full_page=True)

        return f"Capture d'écran sauvegardée : {screenshot_path}"

    except Exception as e:
        return f"Erreur lors de la capture d'écran : {str(e)}"


@tool
def browser_get_content() -> str:
    """Récupère le contenu textuel de la page web actuellement chargée.

    Utilise cette fonction pour lire le texte visible sur la page après
    une navigation ou une interaction. Le contenu est nettoyé du HTML.

    Returns:
        Le texte visible de la page, tronqué à 8000 caractères si nécessaire.
    """
    try:
        page = _get_page()

        # inner_text() retourne uniquement le texte visible de la page
        # (contrairement à textContent qui inclut le texte des éléments cachés)
        content = page.inner_text("body")

        # Troncature pour ne pas saturer le contexte du LLM
        max_length = 8000
        if len(content) > max_length:
            content = content[:max_length] + "\n\n[... contenu tronqué]"

        # Ajout du titre et de l'URL pour le contexte
        title = page.title()
        url = page.url
        return f"Page : {title}\nURL : {url}\n\nContenu :\n{content}"

    except Exception as e:
        return f"Erreur lors de la récupération du contenu : {str(e)}"


@tool
def browser_close() -> str:
    """Ferme le navigateur automatisé et libère les ressources.

    Utilise cette fonction quand tu as terminé toutes les tâches de
    navigation web. Cela libère la mémoire et les processus du navigateur.

    Returns:
        Confirmation de la fermeture.
    """
    global _playwright_instance, _browser, _page

    try:
        # Fermeture en cascade : page → navigateur → Playwright
        if _page and not _page.is_closed():
            _page.close()
        if _browser:
            _browser.close()
        if _playwright_instance:
            _playwright_instance.stop()

        # Réinitialisation des variables globales
        _page = None
        _browser = None
        _playwright_instance = None

        return "Navigateur fermé avec succès."

    except Exception as e:
        return f"Erreur lors de la fermeture du navigateur : {str(e)}"
