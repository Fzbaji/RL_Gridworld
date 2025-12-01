"""
Agent V-Learning (Temporal Difference Learning).
Apprend la fonction de valeur V en interagissant avec l'environnement.
"""

import numpy as np


class VLearningAgent:
    """
    Agent qui utilise l'algorithme V-Learning (TD Learning) pour apprendre.
    Contrairement à Value Iteration, cet agent apprend par expérience.
    """
    
    def __init__(self, n_states, epsilon=0.1, alpha=0.1, gamma=0.99):
        """
        Initialise l'agent V-Learning.
        
        Args:
            n_states (int): Nombre total d'états
            epsilon (float): Probabilité d'exploration (epsilon-greedy)
            alpha (float): Taux d'apprentissage
            gamma (float): Facteur de discount
        """
        self.V = np.zeros(n_states)  # Table de valeurs
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_states = n_states
    
    def choose_action(self, state, env):
        """
        Choisit une action selon une politique epsilon-greedy.
        
        Args:
            state (int): État actuel
            env: L'environnement
        
        Returns:
            int: Action choisie
        """
        # Exploration : action aléatoire
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        
        # Exploitation : meilleure action selon V
        n_actions = env.action_space.n
        action_values = []
        
        for action in range(n_actions):
            next_state, reward = self._simulate_action(env, state, action)
            value = reward + self.gamma * self.V[next_state]
            action_values.append(value)
        
        return np.argmax(action_values)
    
    def _simulate_action(self, env, state, action):
        """
        Simule une action pour évaluer sa valeur (sans modifier l'environnement).
        
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
        
        if env._is_valid_position(new_pos):
            next_pos = new_pos
        else:
            next_pos = pos
        
        next_state = env._pos_to_state(next_pos)
        
        reward = -1
        if next_pos in env.goals:
            reward = 10
        
        return next_state, reward
    
    def update(self, state, reward, next_state, terminated):
        """
        Met à jour la valeur V(s) selon la règle TD(0).
        Formule: V(s) ← V(s) + α[R + γV(s') - V(s)]
        
        Args:
            state (int): État actuel
            reward (float): Récompense reçue
            next_state (int): État suivant
            terminated (bool): Si l'épisode est terminé
        """
        if terminated:
            # Si terminé, V(s') = 0
            td_target = reward
        else:
            td_target = reward + self.gamma * self.V[next_state]
        
        td_error = td_target - self.V[state]
        self.V[state] = self.V[state] + self.alpha * td_error
    
    def train(self, env, n_episodes=1000, verbose=True, visualize_episodes=None):
        """
        Entraîne l'agent en jouant plusieurs épisodes.
        
        Args:
            env: L'environnement
            n_episodes (int): Nombre d'épisodes d'entraînement
            verbose (bool): Afficher les progrès
            visualize_episodes (list): Liste des numéros d'épisodes à visualiser
        
        Returns:
            rewards_history (list): Historique des récompenses totales par épisode
            steps_history (list): Historique du nombre de pas par épisode
        """
        import matplotlib.pyplot as plt
        
        rewards_history = []
        steps_history = []
        
        if visualize_episodes is None:
            visualize_episodes = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 200
            
            # Si on doit visualiser cet épisode
            should_visualize = (episode + 1) in visualize_episodes
            
            if should_visualize:
                plt.ion()
                fig, ax = plt.subplots(figsize=(8, 8))
                path = []
            
            for step in range(max_steps):
                # Visualiser l'état actuel
                if should_visualize:
                    self._visualize_step(env, state, ax, episode + 1, step, total_reward)
                    path.append(env._state_to_pos(state))
                
                # Choisir et exécuter une action
                action = self.choose_action(state, env)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Mettre à jour V(s)
                self.update(state, reward, next_state, terminated or truncated)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if terminated or truncated:
                    if should_visualize:
                        # Afficher l'état final
                        self._visualize_step(env, state, ax, episode + 1, step + 1, 
                                           total_reward, final=True)
                    break
            
            if should_visualize:
                plt.ioff()
                plt.close(fig)
            
            rewards_history.append(total_reward)
            steps_history.append(steps)
            
            # Afficher les progrès
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                avg_steps = np.mean(steps_history[-100:])
                print(f"Épisode {episode + 1}/{n_episodes} - "
                      f"Récompense moyenne: {avg_reward:.2f} - "
                      f"Pas moyens: {avg_steps:.2f}")
        
        if verbose:
            print(f"\n✓ Entraînement terminé après {n_episodes} épisodes")
        
        return rewards_history, steps_history
    
    def _visualize_step(self, env, state, ax, episode, step, total_reward, final=False):
        """
        Visualise un pas d'apprentissage.
        
        Args:
            env: L'environnement
            state: État actuel
            ax: Axes matplotlib
            episode: Numéro d'épisode
            step: Numéro du pas
            total_reward: Récompense totale accumulée
            final: Si c'est le dernier pas
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        ax.clear()
        
        # Créer la grille de base
        grid = np.ones((env.size, env.size))
        
        for obs in env.obstacles:
            grid[obs] = 0.3
        
        for goal in env.goals:
            grid[goal] = 0.7
        
        ax.imshow(grid, cmap='YlGn', vmin=0, vmax=1, alpha=0.6)
        
        # Dessiner les obstacles
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
                                       fill=True, color='gray', alpha=0.8))
            ax.text(obs[1], obs[0], 'X', ha='center', va='center', 
                   fontsize=20, fontweight='bold', color='red')
        
        # Dessiner les goals
        for goal in env.goals:
            ax.add_patch(plt.Circle((goal[1], goal[0]), 0.35, 
                                   fill=True, color='gold', alpha=0.9))
            ax.text(goal[1], goal[0], '★', ha='center', va='center', 
                   fontsize=25, color='darkgreen')
        
        # Dessiner l'agent
        current_pos = env._state_to_pos(state)
        ax.add_patch(plt.Circle((current_pos[1], current_pos[0]), 0.3, 
                               fill=True, color='purple', alpha=0.9, zorder=10))
        ax.text(current_pos[1], current_pos[0], '●', ha='center', va='center', 
               fontsize=30, color='white', fontweight='bold')
        
        # Marquer le départ
        ax.scatter(env.start_pos[1], env.start_pos[0], marker='s', 
                  s=500, c='lightgreen', alpha=0.5, zorder=1)
        
        # Grille
        ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        
        # Labels
        ax.set_xlabel('Colonne', fontsize=12)
        ax.set_ylabel('Ligne', fontsize=12)
        title = f'V-Learning - Épisode {episode} - Pas: {step}'
        if final:
            title += f' - ✓ Terminé! Récompense: {total_reward:.1f}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Info box
        info_text = f'Récompense totale: {total_reward:.1f}\nPas: {step}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.draw()
        
        if final:
            plt.pause(2.0)  # Pause plus longue à la fin
        else:
            plt.pause(0.3)  # Pause courte pendant l'épisode
    
    def act(self, state, env):
        """
        Sélectionne l'action optimale (sans exploration) après l'entraînement.
        
        Args:
            state (int): État actuel
            env: L'environnement
        
        Returns:
            int: Action optimale
        """
        n_actions = env.action_space.n
        action_values = []
        
        for action in range(n_actions):
            next_state, reward = self._simulate_action(env, state, action)
            value = reward + self.gamma * self.V[next_state]
            action_values.append(value)
        
        return np.argmax(action_values)
