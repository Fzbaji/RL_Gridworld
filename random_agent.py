"""
Agent aléatoire pour le GridWorld.
Choisit des actions au hasard.
"""


class RandomAgent:
    """
    Agent qui sélectionne des actions aléatoires.
    """
    
    def __init__(self):
        """Initialise l'agent aléatoire."""
        pass
    
    def act(self, observation, env):
        """
        Sélectionne une action aléatoire.
        
        Args:
            observation: L'état actuel (non utilisé pour l'agent aléatoire)
            env: L'environnement (pour accéder à action_space)
        
        Returns:
            int: Action aléatoire sélectionnée
        """
        return env.action_space.sample()
