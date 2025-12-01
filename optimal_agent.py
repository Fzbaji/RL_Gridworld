"""
Agent optimal utilisant l'algorithme Value Iteration.
Calcule la politique optimale pour résoudre le GridWorld.
"""

import numpy as np


class ValueIterationAgent:
    """
    Agent qui utilise l'algorithme Value Iteration pour trouver la politique optimale.
    """
    
    def __init__(self):
        """Initialise l'agent Value Iteration."""
        self.V_table = None
    
    def train(self, env, gamma=0.99, theta=1e-6):
        """
        Entraîne l'agent en utilisant l'algorithme Value Iteration.
        
        Args:
            env: L'environnement GridWorld
            gamma (float): Facteur de discount
            theta (float): Seuil de convergence
        
        Returns:
            V_table (np.array): Table des valeurs des états
            deltas (list): Historique des erreurs (delta) pour chaque itération
        """
        # Initialiser la table de valeurs
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        V = np.zeros(n_states)
        
        deltas = []
        iteration = 0
        
        while True:
            delta = 0
            iteration += 1
            
            # Pour chaque état
            for state in range(n_states):
                pos = env._state_to_pos(state)
                
                # Si l'état est un goal, sa valeur reste 0 (état terminal)
                if pos in env.goals:
                    continue
                
                # Si l'état est un obstacle, sa valeur reste 0 (non accessible)
                if pos in env.obstacles:
                    continue
                
                v = V[state]
                
                # Calculer la valeur maximale sur toutes les actions
                action_values = []
                for action in range(n_actions):
                    # Simuler l'action
                    next_state, reward = self._simulate_action(env, state, action)
                    value = reward + gamma * V[next_state]
                    action_values.append(value)
                
                # Mettre à jour la valeur de l'état
                V[state] = max(action_values)
                
                # Calculer le changement
                delta = max(delta, abs(v - V[state]))
            
            deltas.append(delta)
            
            # Vérifier la convergence
            if delta < theta:
                print(f"Convergence atteinte après {iteration} itérations")
                break
        
        self.V_table = V
        return V, deltas
    
    def _simulate_action(self, env, state, action):
        """
        Simule une action sans modifier l'état de l'environnement.
        
        Args:
            env: L'environnement
            state (int): État actuel
            action (int): Action à simuler
        
        Returns:
            next_state (int): État suivant
            reward (float): Récompense obtenue
        """
        pos = env._state_to_pos(state)
        direction = env.action_to_direction[action]
        new_row = pos[0] + direction[0]
        new_col = pos[1] + direction[1]
        new_pos = (new_row, new_col)
        
        # Vérifier si la nouvelle position est valide
        if env._is_valid_position(new_pos):
            next_pos = new_pos
        else:
            next_pos = pos  # Reste sur place
        
        next_state = env._pos_to_state(next_pos)
        
        # Calculer la récompense
        reward = -1
        if next_pos in env.goals:
            reward = 10
        
        return next_state, reward
    
    def act(self, state, env):
        """
        Sélectionne l'action optimale basée sur la politique greedy.
        
        Args:
            state (int): État actuel
            env: L'environnement
        
        Returns:
            int: Action optimale
        """
        if self.V_table is None:
            raise ValueError("L'agent doit être entraîné avant de pouvoir agir!")
        
        n_actions = env.action_space.n
        action_values = []
        
        # Calculer la valeur de chaque action
        for action in range(n_actions):
            next_state, reward = self._simulate_action(env, state, action)
            value = reward + 0.99 * self.V_table[next_state]  # Utilise gamma par défaut
            action_values.append(value)
        
        # Retourner l'action avec la valeur maximale
        return np.argmax(action_values)
