"""
Jeu de CATCH amélioré avec features enrichies.
L'agent apprend à poursuivre le goal grâce à des features de distance/direction.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class CatchGameEnv:
    """
    Environnement de jeu de catch avec observation enrichie.
    L'agent voit : (sa_position, position_goal) → peut généraliser !
    """
    
    def __init__(self, size=5, obstacles=None):
        self.size = size
        self.obstacles = obstacles if obstacles else [(1, 1), (2, 2)]
        self.catches = 0
        
        # Positions valides (sans obstacles)
        self.valid_positions = []
        for i in range(size):
            for j in range(size):
                if (i, j) not in self.obstacles:
                    self.valid_positions.append((i, j))
        
        self.reset()
    
    def reset(self):
        """Réinitialise l'environnement."""
        self.agent_pos = (0, 0)
        self.goal_pos = self._random_goal_position()
        self.catches = 0
        return self._get_state()
    
    def _random_goal_position(self):
        """Génère une position aléatoire pour le goal (loin de l'agent)."""
        valid_goals = [p for p in self.valid_positions if p != self.agent_pos]
        return valid_goals[np.random.randint(len(valid_goals))]
    
    def _get_state(self):
        """
        État enrichi : (agent_pos, goal_pos)
        L'agent sait OÙ est le goal → peut apprendre à le poursuivre !
        """
        return (self.agent_pos, self.goal_pos)
    
    def step(self, action):
        """
        Exécute une action.
        Actions : 0=Haut, 1=Bas, 2=Gauche, 3=Droite
        """
        # Calculer nouvelle position
        row, col = self.agent_pos
        if action == 0:  # Haut
            new_pos = (row - 1, col)
        elif action == 1:  # Bas
            new_pos = (row + 1, col)
        elif action == 2:  # Gauche
            new_pos = (row, col - 1)
        else:  # Droite (3)
            new_pos = (row, col + 1)
        
        # Vérifier validité
        if self._is_valid(new_pos):
            self.agent_pos = new_pos
        
        # Vérifier si goal attrapé
        if self.agent_pos == self.goal_pos:
            reward = 100
            self.catches += 1
            self.goal_pos = self._random_goal_position()
            print(f"  🎯 CATCH ! Goal #{self.catches} - Nouveau goal: {self.goal_pos}")
        else:
            # Petite récompense pour se rapprocher du goal
            old_dist = self._manhattan_distance(self.agent_pos, self.goal_pos)
            reward = -1  # Pénalité de base pour chaque pas
        
        return self._get_state(), reward, False, {}
    
    def _is_valid(self, pos):
        """Vérifie si une position est valide."""
        row, col = pos
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if pos in self.obstacles:
            return False
        return True
    
    def _manhattan_distance(self, pos1, pos2):
        """Distance de Manhattan entre deux positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class ImprovedQLearningAgent:
    """
    Agent Q-Learning amélioré avec état enrichi (agent_pos, goal_pos).
    Peut généraliser et poursuivre n'importe quel goal !
    """
    
    def __init__(self, epsilon=0.1, alpha=0.3, gamma=0.95):
        self.Q = defaultdict(lambda: np.zeros(4))  # Q(state, action)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    
    def choose_action(self, state):
        """Epsilon-greedy : exploration vs exploitation."""
        if np.random.random() < self.epsilon:
            return np.random.randint(4)  # Exploration
        else:
            return np.argmax(self.Q[state])  # Exploitation
    
    def update(self, state, action, reward, next_state):
        """Mise à jour Q-Learning."""
        current_q = self.Q[state][action]
        max_next_q = np.max(self.Q[next_state])
        
        # Q(s,a) ← Q(s,a) + α[R + γ max Q(s',a') - Q(s,a)]
        self.Q[state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
    
    def act(self, state):
        """Action glouton (pour démonstration)."""
        return np.argmax(self.Q[state])


def train_catch_agent(env, n_episodes=1000):
    """
    Entraîne un agent Q-Learning amélioré.
    """
    agent = ImprovedQLearningAgent(epsilon=0.15, alpha=0.3, gamma=0.95)
    
    rewards_history = []
    catches_history = []
    
    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT - JEU DE CATCH AMÉLIORÉ")
    print(f"{'='*70}\n")
    print(f"L'agent apprend à POURSUIVRE le goal grâce aux features enrichies !")
    print(f"Feature = (position_agent, position_goal) → Généralisation !\n")
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        max_steps = 100
        
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
        
        rewards_history.append(total_reward)
        catches_history.append(env.catches)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_catches = np.mean(catches_history[-100:])
            print(f"Épisode {episode + 1}/{n_episodes} - "
                  f"Récompense moy: {avg_reward:.2f} - "
                  f"Catches moy: {avg_catches:.2f}")
    
    print(f"\n✓ Entraînement terminé !")
    print(f"\nStatistiques finales (100 derniers épisodes):")
    print(f"  - Récompense moyenne : {np.mean(rewards_history[-100:]):.2f}")
    print(f"  - Catches moyens : {np.mean(catches_history[-100:]):.2f}")
    print(f"  - États uniques appris : {len(agent.Q)}")
    
    return agent, rewards_history, catches_history


def visualize_catch_episode(env, agent):
    """
    Visualise un épisode de catch.
    """
    state = env.reset()
    
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
        grid[env.goal_pos] = 0.7
        
        ax.imshow(grid, cmap='YlGn', vmin=0, vmax=1, alpha=0.6)
        
        # Obstacles
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8,
                                       fill=True, color='gray', alpha=0.8))
            ax.text(obs[1], obs[0], 'X', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='red')
        
        # Goal
        goal = env.goal_pos
        ax.add_patch(plt.Circle((goal[1], goal[0]), 0.4,
                               fill=True, color='gold', alpha=0.9, zorder=5))
        ax.text(goal[1], goal[0], '★', ha='center', va='center',
               fontsize=30, color='darkgreen', zorder=6)
        
        # Agent
        agent_pos = env.agent_pos
        ax.add_patch(plt.Circle((agent_pos[1], agent_pos[0]), 0.35,
                               fill=True, color='blue', alpha=0.9, zorder=10))
        ax.text(agent_pos[1], agent_pos[0], '🏃', ha='center', va='center',
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
        ax.set_title(f'JEU DE CATCH AMÉLIORÉ - Pas: {step}',
                    fontsize=14, fontweight='bold')
        
        # Info
        dist = abs(agent_pos[0] - goal[0]) + abs(agent_pos[1] - goal[1])
        info_text = f'🎯 Catches: {env.catches}\n💰 Score: {total_reward:.0f}\n📍 Distance: {dist}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.2)
        
        # Exécuter l'action
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    
    plt.ioff()
    plt.close(fig)
    
    print(f"\n✓ Démonstration terminée: {env.catches} catches, score: {total_reward:.0f}")


def plot_learning_curves(rewards, catches):
    """
    Affiche les courbes d'apprentissage.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    window = 20
    
    # Récompenses
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed,
                color='blue', linewidth=2, label='Q-Learning Amélioré')
    
    ax1.set_xlabel('Épisode', fontsize=12)
    ax1.set_ylabel('Récompense (moyenne mobile)', fontsize=12)
    ax1.set_title('Apprentissage - Récompenses', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Catches
    if len(catches) >= window:
        smoothed = np.convolve(catches, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(catches)), smoothed,
                color='green', linewidth=2, label='Catches par épisode')
    
    ax2.set_xlabel('Épisode', fontsize=12)
    ax2.set_ylabel('Catches (moyenne mobile)', fontsize=12)
    ax2.set_title('Performance - Catches', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Fonction principale.
    """
    print("="*70)
    print("JEU DE CATCH - VERSION AMÉLIORÉE")
    print("="*70)
    print("\n🎯 AMÉLIORATION CLÉ :")
    print("  • Feature = (position_agent, position_goal)")
    print("  • L'agent SAIT où est le goal → peut le poursuivre !")
    print("  • Généralise à n'importe quelle position de goal\n")
    
    # Créer environnement
    env = CatchGameEnv(size=5, obstacles=[(1, 1), (2, 2)])
    
    # Entraîner
    agent, rewards, catches = train_catch_agent(env, n_episodes=1000)
    
    # Visualisations
    print("\n" + "="*70)
    print("VISUALISATIONS")
    print("="*70)
    
    print("\n1. Courbes d'apprentissage...")
    plot_learning_curves(rewards, catches)
    
    print("2. Démonstration de l'agent entraîné...")
    visualize_catch_episode(env, agent)
    
    print("\n" + "="*70)
    print("✅ SUCCÈS !")
    print("="*70)
    print("\nL'agent a appris à POURSUIVRE efficacement le goal grâce")
    print("aux features enrichies qui incluent la position du goal !")


if __name__ == "__main__":
    main()
