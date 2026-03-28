"""
ProjectX - Outil d'Envoi d'Emails (SMTP)
Permet à l'agent d'envoyer des emails via un serveur SMTP configuré.
Supporte TLS pour la connexion sécurisée.

Les identifiants SMTP doivent être configurés dans config.py ou via des
variables d'environnement (SMTP_SERVER, SMTP_PORT, SMTP_USERNAME,
SMTP_PASSWORD) pour des raisons de sécurité.

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain_core.tools import tool

from config import SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Envoie un email via le serveur SMTP configuré.

    Utilise cette fonction pour envoyer des emails à une adresse donnée.
    Le serveur SMTP et les identifiants doivent être configurés dans
    config.py ou via les variables d'environnement correspondantes.

    Args:
        to: Adresse email du destinataire.
            Exemple : "destinataire@example.com".
        subject: Objet de l'email.
            Exemple : "Rapport quotidien ProjectX".
        body: Corps de l'email en texte brut.
            Exemple : "Voici le résumé des actions effectuées aujourd'hui...".

    Returns:
        Confirmation de l'envoi avec les détails, ou message d'erreur
        détaillé si l'envoi échoue (configuration manquante, auth, etc.).
    """
    # --- Vérification de la configuration SMTP ---
    # On vérifie que les paramètres essentiels sont définis avant de tenter
    # la connexion pour donner un message d'erreur clair à l'agent
    if not SMTP_SERVER:
        return (
            "Erreur : le serveur SMTP n'est pas configuré. "
            "Définissez la variable d'environnement SMTP_SERVER ou "
            "modifiez config.py (ex: SMTP_SERVER='smtp.gmail.com')."
        )

    if not SMTP_USERNAME or not SMTP_PASSWORD:
        return (
            "Erreur : les identifiants SMTP ne sont pas configurés. "
            "Définissez les variables d'environnement SMTP_USERNAME et "
            "SMTP_PASSWORD pour pouvoir envoyer des emails."
        )

    try:
        # --- Construction du message email ---
        # On utilise MIMEMultipart pour permettre l'évolution future
        # vers des emails HTML ou avec pièces jointes
        msg = MIMEMultipart()
        msg["From"] = SMTP_FROM_EMAIL  # Expéditeur
        msg["To"] = to  # Destinataire
        msg["Subject"] = subject  # Objet

        # Ajout du corps en texte brut (UTF-8 pour les accents français)
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # --- Connexion au serveur SMTP et envoi ---
        # On utilise SMTP standard (port 587) avec starttls pour le chiffrement
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            # EHLO : identification auprès du serveur SMTP
            server.ehlo()

            # STARTTLS : passage en connexion chiffrée TLS
            # Protège les identifiants et le contenu de l'email
            server.starttls()
            server.ehlo()  # Re-identification après TLS

            # Authentification avec les identifiants configurés
            server.login(SMTP_USERNAME, SMTP_PASSWORD)

            # Envoi effectif de l'email
            server.send_message(msg)

        return (
            f"Email envoyé avec succès !\n"
            f"  De : {SMTP_FROM_EMAIL}\n"
            f"  À  : {to}\n"
            f"  Objet : {subject}"
        )

    except smtplib.SMTPAuthenticationError:
        return (
            "Erreur d'authentification SMTP : identifiants incorrects. "
            "Vérifiez SMTP_USERNAME et SMTP_PASSWORD. "
            "Pour Gmail, utilisez un mot de passe d'application."
        )
    except smtplib.SMTPConnectError:
        return (
            f"Erreur de connexion au serveur SMTP {SMTP_SERVER}:{SMTP_PORT}. "
            "Vérifiez que le serveur et le port sont corrects."
        )
    except smtplib.SMTPException as e:
        return f"Erreur SMTP lors de l'envoi : {str(e)}"
    except Exception as e:
        return f"Erreur inattendue lors de l'envoi de l'email : {str(e)}"
