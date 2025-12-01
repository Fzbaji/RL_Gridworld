"""
Expérience : Jeu de "CATCH" - L'agent poursuit le goal qui se déplace.
L'agent doit attraper le goal qui se déplace à chaque fois qu'il est attrapé.
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from v_learning_agent import VLearningAgent
from q_learning_agent import QLearningAgent


class CatchGoalEnv(GridWorldEnv):
    """
    Environnement "Catch" : L'agent doit attraper un goal qui se déplace.
    Quand l'agent attrape le goal, il gagne des points et le goal se déplace ailleurs.
    """
    
    def __init__(self, size=5, start=(0, 0), obstacles=None, initial_goal=(4, 4)):
        """
        Args:
            size: Taille de la grille
            start: Position de départ de l'agent
            obstacles: Liste des obstacles
            initial_goal: Position initiale du goal
        """
        self.initial_goal = initial_goal
        self.catches = 0  # Nombre de fois que l'agent a attrapé le goal
        super().__init__(size, start, obstacles, [initial_goal])
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement."""
        observation, info = super().reset(seed, options)
        self.goals = [self.initial_goal]
        self.catches = 0
        return observation, info
    
    def step(self, action):
        """
        Exécute un pas. Si l'agent atteint le goal:
        - Il reçoit une récompense
        - Le goal se déplace vers une nouvelle position
        - L'épisode continue
        """
        # Exécuter l'action
        old_pos = self.current_pos
        next_pos = self._get_next_position(action)
        
        # Vérifier si la position est valide
        if self._is_valid_position(next_pos):
            self.current_pos = next_pos
        
        # Nouvelle observation
        observation = self._pos_to_state(self.current_pos)
        
        # Vérifier si l'agent a attrapé le goal
        if self.current_pos in self.goals:
            reward = 100  # Grosse récompense pour avoir attrapé le goal !
            self.catches += 1
            # Déplacer le goal vers une nouvelle position
            self._move_goal_to_random_position()
            terminated = False  # L'épisode continue !
        else:
            reward = -1  # Pénalité pour chaque pas (encourager à attraper vite)
            terminated = False
        
        truncated = False
        info = {"catches": self.catches}
        
        return observation, reward, terminated, truncated, info
    
    def _move_goal_to_random_position(self):
        """Déplace le goal vers une position aléatoire valide (loin de l'agent)."""
        # Trouver toutes les positions valides
        valid_positions = []
        for i in range(self.size):
            for j in range(self.size):
                pos = (i, j)
                # Position valide si : pas obstacle, pas position de l'agent
                if pos not in self.obstacles and pos != self.current_pos:
                    valid_positions.append(pos)
        
        if valid_positions:
            # Choisir une position aléatoire
            new_goal = valid_positions[np.random.randint(len(valid_positions))]
            self.goals = [new_goal]
            print(f"  🎯 Goal attrapé ! Nouveau goal: {new_goal} (Total: {self.catches})")
    
    def _get_next_position(self, action):
        """Calcule la prochaine position en fonction de l'action."""
        row, col = self.current_pos
        if action == 0:  # Haut
            return (row - 1, col)
        elif action == 1:  # Bas
            return (row + 1, col)
        elif action == 2:  # Gauche
            return (row, col - 1)
        elif action == 3:  # Droite
            return (row, col + 1)
        return self.current_pos
    
    def _is_valid_position(self, pos):
        """Vérifie si une position est valide."""
        row, col = pos
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if pos in self.obstacles:
            return False
        return True


def train_with_catch_goal(agent_class, env, n_episodes=500, agent_name="Agent"):
    """
    Entraîne un agent à attraper le goal mobile.
    
    Args:
        agent_class: Classe de l'agent (VLearningAgent ou QLearningAgent)
        env: L'environnement Catch
        n_episodes (int): Nombre d'épisodes
        agent_name (str): Nom de l'agent pour l'affichage
    
    Returns:
        agent, rewards_history, catches_history
    """
    # Créer l'agent
    if agent_class == VLearningAgent:
        agent = VLearningAgent(
            n_states=env.observation_space.n,
            epsilon=0.2,  # Exploration pour trouver le goal
            alpha=0.2,    # Apprentissage rapide
            gamma=0.95
        )
    else:  # QLearningAgent
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            epsilon=0.2,
            alpha=0.2,
            gamma=0.95
        )
    
    rewards_history = []
    catches_history = []
    
    print(f"\n{'='*70}")
    print(f"Entraînement de {agent_name} - JEU DE CATCH")
    print(f"{'='*70}\n")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        max_steps = 100
        
        for step in range(max_steps):
            # Choisir et exécuter une action
            if agent_class == VLearningAgent:
                action = agent.choose_action(state, env)
            else:
                action = agent.choose_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Mettre à jour l'agent
            if agent_class == VLearningAgent:
                agent.update(state, reward, next_state, False)  # Jamais terminated
            else:
                agent.update(state, action, reward, next_state, False)
            
            total_reward += reward
            state = next_state
        
        rewards_history.append(total_reward)
        catches_history.append(env.catches)
        
        # Afficher les progrès
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_catches = np.mean(catches_history[-50:])
            print(f"Épisode {episode + 1}/{n_episodes} - "
                  f"Récompense moy: {avg_reward:.2f} - "
                  f"Catches moy: {avg_catches:.2f}")
    
    print(f"\n✓ Entraînement terminé !")
    return agent, rewards_history, catches_history


def visualize_catch_episode(env, agent, agent_name="Agent"):
    """
    Visualise un épisode complet du jeu de catch.
    
    Args:
        env: L'environnement
        agent: L'agent entraîné
        agent_name (str): Nom pour le titre
    """
    state, _ = env.reset()
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    max_steps = 100
    total_reward = 0
    
    for step in range(max_steps):
        ax.clear()
        
        # Grille de base
        grid = np.ones((env.size, env.size))
        for obs in env.obstacles:
            grid[obs] = 0.3
        for goal in env.goals:
            grid[goal] = 0.7
        
        ax.imshow(grid, cmap='YlGn', vmin=0, vmax=1, alpha=0.6)
        
        # Obstacles
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8,
                                       fill=True, color='gray', alpha=0.8))
            ax.text(obs[1], obs[0], 'X', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='red')
        
        # Goal (étoile animée !)
        for goal in env.goals:
            ax.add_patch(plt.Circle((goal[1], goal[0]), 0.4,
                                   fill=True, color='gold', alpha=0.9, zorder=5))
            ax.text(goal[1], goal[0], '★', ha='center', va='center',
                   fontsize=30, color='darkgreen', zorder=6)
        
        # Agent (poursuiveur !)
        current_pos = env._state_to_pos(state)
        color = 'purple' if 'V-Learning' in agent_name else 'orange'
        ax.add_patch(plt.Circle((current_pos[1], current_pos[0]), 0.3,
                               fill=True, color=color, alpha=0.9, zorder=10))
        ax.text(current_pos[1], current_pos[0], '🏃', ha='center', va='center',
               fontsize=25)
        
        # Grille
        ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        
        # Labels
        ax.set_xlabel('Colonne', fontsize=12)
        ax.set_ylabel('Ligne', fontsize=12)
        ax.set_title(f'{agent_name} - JEU DE CATCH - Pas: {step}',
                    fontsize=14, fontweight='bold')
        
        # Info
        info_text = f'🎯 Catches: {env.catches}\n💰 Récompense: {total_reward:.0f}\n📍 Goal: {env.goals[0]}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.2)
        
        # Exécuter l'action
        if isinstance(agent, VLearningAgent):
            action = agent.act(state, env)
        else:
            action = agent.act(state)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
    
    plt.ioff()
    plt.close(fig)
    
    print(f"\n✓ Épisode terminé: {env.catches} goals attrapés, récompense totale: {total_reward:.0f}")


def plot_catch_results(v_rewards, v_catches, q_rewards, q_catches):
    """
    Compare les résultats des deux agents pour le jeu de catch.
    
    Args:
        v_rewards: Récompenses V-Learning
        v_catches: Catches V-Learning
        q_rewards: Récompenses Q-Learning
        q_catches: Catches Q-Learning
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    window = 20
    episodes = range(len(v_rewards))
    
    # Graphique des récompenses
    if len(v_rewards) >= window:
        v_smooth = np.convolve(v_rewards, np.ones(window)/window, mode='valid')
        q_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
        
        ax1.plot(range(window-1, len(v_rewards)), v_smooth,
                color='purple', linewidth=2, label='V-Learning')
        ax1.plot(range(window-1, len(q_rewards)), q_smooth,
                color='orange', linewidth=2, label='Q-Learning')
    
    ax1.set_xlabel('Épisode', fontsize=12)
    ax1.set_ylabel('Récompense totale (moyenne mobile)', fontsize=12)
    ax1.set_title('JEU DE CATCH - Récompenses', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Graphique des catches
    if len(v_catches) >= window:
        v_catches_smooth = np.convolve(v_catches, np.ones(window)/window, mode='valid')
        q_catches_smooth = np.convolve(q_catches, np.ones(window)/window, mode='valid')
        
        ax2.plot(range(window-1, len(v_catches)), v_catches_smooth,
                color='purple', linewidth=2, label='V-Learning')
        ax2.plot(range(window-1, len(q_catches)), q_catches_smooth,
                color='orange', linewidth=2, label='Q-Learning')
    
    ax2.set_xlabel('Épisode', fontsize=12)
    ax2.set_ylabel('Goals attrapés par épisode', fontsize=12)
    ax2.set_title('JEU DE CATCH - Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Fonction principale pour l'expérience jeu de CATCH.
    """
    print("="*70)
    print("EXPÉRIENCE : JEU DE CATCH (POURSUITE DU GOAL)")
    print("="*70)
    print("\nConcept: L'agent doit ATTRAPER le goal.")
    print("Quand l'agent attrape le goal:")
    print("  → +100 points")
    print("  → Le goal se déplace ailleurs")
    print("  → L'agent continue à le poursuivre !\n")
    
    # Créer les environnements
    env_v = CatchGoalEnv(
        size=5,
        start=(0, 0),
        obstacles=[(1, 1), (2, 2)],
        initial_goal=(4, 4)
    )
    
    env_q = CatchGoalEnv(
        size=5,
        start=(0, 0),
        obstacles=[(1, 1), (2, 2)],
        initial_goal=(4, 4)
    )
    
    print(f"Environnement: Grille {env_v.size}x{env_v.size}")
    print(f"Goal initial: {env_v.initial_goal}")
    print(f"Obstacles: {env_v.obstacles}")
    print(f"Récompense par catch: +100")
    print(f"Pénalité par pas: -1")
    
    # Entraîner V-Learning
    v_agent, v_rewards, v_catches = train_with_catch_goal(
        VLearningAgent, env_v, n_episodes=500, agent_name="V-Learning"
    )
    
    # Entraîner Q-Learning
    q_agent, q_rewards, q_catches = train_with_catch_goal(
        QLearningAgent, env_q, n_episodes=500, agent_name="Q-Learning"
    )
    
    # Statistiques finales
    print("\n" + "="*70)
    print("STATISTIQUES FINALES (50 derniers épisodes)")
    print("="*70)
    print(f"\nV-Learning:")
    print(f"  - Récompense moyenne: {np.mean(v_rewards[-50:]):.2f}")
    print(f"  - Catches moyens: {np.mean(v_catches[-50:]):.2f}")
    
    print(f"\nQ-Learning:")
    print(f"  - Récompense moyenne: {np.mean(q_rewards[-50:]):.2f}")
    print(f"  - Catches moyens: {np.mean(q_catches[-50:]):.2f}")
    
    # Comparaison graphique
    print("\n" + "="*70)
    print("VISUALISATIONS")
    print("="*70)
    
    print("\n1. Comparaison des performances...")
    plot_catch_results(v_rewards, v_catches, q_rewards, q_catches)
    
    print("2. Démonstration V-Learning - Jeu de Catch...")
    visualize_catch_episode(env_v, v_agent, "V-Learning")
    
    print("3. Démonstration Q-Learning - Jeu de Catch...")
    visualize_catch_episode(env_q, q_agent, "Q-Learning")
    
    print("\n" + "="*70)
    print("EXPÉRIENCE TERMINÉE !")
    print("="*70)
    print("\nOBSERVATIONS:")
    print("• L'agent apprend à POURSUIVRE et ATTRAPER le goal")
    print("• Chaque catch = +100 points, le goal réapparaît ailleurs")
    print("• Environnement dynamique : le goal change à chaque catch")
    print("• L'agent doit être rapide et efficace pour maximiser les catches")


if __name__ == "__main__":
    main()
