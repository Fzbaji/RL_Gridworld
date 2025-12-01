"""
GridWorld Environment pour l'apprentissage par renforcement.
Hérite de gym.Env pour être compatible avec Gymnasium.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridWorldEnv(gym.Env):
    """
    Environnement GridWorld avec obstacles et objectifs.
    
    Actions:
        0: Haut
        1: Bas
        2: Gauche
        3: Droite
    """
    
    def __init__(self, size=5, start=(0, 0), obstacles=None, goals=None):
        """
        Initialise l'environnement GridWorld.
        
        Args:
            size (int): Taille de la grille (size x size)
            start (tuple): Position de départ (row, col)
            obstacles (list): Liste de tuples représentant les obstacles
            goals (list): Liste de tuples représentant les objectifs
        """
        super(GridWorldEnv, self).__init__()
        
        self.size = size
        self.start_pos = start
        self.obstacles = obstacles if obstacles is not None else []
        self.goals = goals if goals is not None else []
        
        # Définition des espaces d'observation et d'action
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)  # 4 directions
        
        # Mappage des actions vers les déplacements
        self.action_to_direction = {
            0: (-1, 0),  # Haut
            1: (1, 0),   # Bas
            2: (0, -1),  # Gauche
            3: (0, 1)    # Droite
        }
        
        # État courant
        self.current_pos = None
        self.reset()
    
    def set_goal(self, new_goal):
        """
        Change la position du goal.
        
        Args:
            new_goal (tuple ou list): Nouvelle position du goal (row, col) ou liste de positions
        """
        if isinstance(new_goal, tuple):
            self.goals = [new_goal]
        else:
            self.goals = new_goal
    
    def _pos_to_state(self, pos):
        """Convertit une position (row, col) en état (index unique)."""
        return pos[0] * self.size + pos[1]
    
    def _state_to_pos(self, state):
        """Convertit un état (index) en position (row, col)."""
        return (state // self.size, state % self.size)
    
    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement à l'état initial.
        
        Returns:
            observation (int): État initial
            info (dict): Informations supplémentaires
        """
        super().reset(seed=seed)
        self.current_pos = self.start_pos
        observation = self._pos_to_state(self.current_pos)
        info = {}
        return observation, info
    
    def step(self, action):
        """
        Exécute une action dans l'environnement.
        
        Args:
            action (int): Action à exécuter (0-3)
        
        Returns:
            observation (int): Nouvel état
            reward (float): Récompense obtenue
            terminated (bool): Si l'épisode est terminé
            truncated (bool): Si l'épisode est tronqué
            info (dict): Informations supplémentaires
        """
        # Calculer la nouvelle position
        direction = self.action_to_direction[action]
        new_row = self.current_pos[0] + direction[0]
        new_col = self.current_pos[1] + direction[1]
        new_pos = (new_row, new_col)
        
        # Vérifier si la nouvelle position est valide
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
        # Sinon, l'agent reste sur place (tape un mur ou un obstacle)
        
        # Calculer la récompense
        reward = -1  # Pénalité pour chaque pas
        terminated = False
        
        if self.current_pos in self.goals:
            reward = 10  # Récompense pour atteindre le goal
            terminated = True
        
        observation = self._pos_to_state(self.current_pos)
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def _is_valid_position(self, pos):
        """
        Vérifie si une position est valide (dans la grille et pas un obstacle).
        
        Args:
            pos (tuple): Position (row, col)
        
        Returns:
            bool: True si valide, False sinon
        """
        row, col = pos
        
        # Vérifier les limites de la grille
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        
        # Vérifier les obstacles
        if pos in self.obstacles:
            return False
        
        return True
    
    def render(self):
        """Affiche l'état actuel de la grille (optionnel)."""
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        
        for obs in self.obstacles:
            grid[obs] = 'X'
        
        for goal in self.goals:
            grid[goal] = 'G'
        
        grid[self.current_pos] = 'A'
        
        print("\n".join([" ".join(row) for row in grid]))
        print()
