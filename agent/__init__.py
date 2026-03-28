"""
ProjectX - Package Agent
Ce package contient le cœur de l'agent AI : le graphe LangGraph qui
orchestre le raisonnement, la mémoire persistante, et le système
d'auto-apprentissage par réflexion.

Modules :
- graph.py    : Graphe LangGraph principal (boucle ReAct enrichie)
- memory.py   : Gestion de la mémoire persistante (SQLite + Store)
- learning.py : Noeuds de réflexion et de rappel (auto-apprentissage)

Auteur  : ProfesseurFalken
Contact : wojcikej@orange.fr
GitHub  : https://github.com/ProfesseurFalken/ProjectX
Date    : 2026-03-28
"""

from agent.graph import create_agent_graph
