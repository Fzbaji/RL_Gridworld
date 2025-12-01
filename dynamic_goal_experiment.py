"""
Script pour tester V-Learning avec un goal qui change de position.
Démontre la capacité d'adaptation de l'agent.
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from v_learning_agent import VLearningAgent


def plot_learning_with_goal_changes(rewards_history, steps_history, goal_changes):
    """
    Affiche les courbes d'apprentissage avec indication des changements de goal.
    
    Args:
        rewards_history (list): Historique des récompenses
        steps_history (list): Historique du nombre de pas
        goal_changes (dict): Episodes où le goal a changé {episode: nouvelle_position}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    episodes = range(len(rewards_history))
    
    # Graphique des récompenses
    ax1.plot(episodes, rewards_history, alpha=0.5, color='blue', linewidth=0.5)
    
    # Moyenne mobile
    window = 50
    if len(rewards_history) >= window:
        rewards_smooth = np.convolve(rewards_history, 
                                     np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards_history)), rewards_smooth, 
                color='red', linewidth=2, label=f'Moyenne mobile ({window})')
    
    # Marquer les changements de goal
    for ep, goal_pos in goal_changes.items():
        ax1.axvline(x=ep, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(ep, ax1.get_ylim()[1]*0.9, f'Goal→{goal_pos}', 
                rotation=90, va='top', fontsize=9, color='green', fontweight='bold')
    
    ax1.set_xlabel('Épisode', fontsize=12)
    ax1.set_ylabel('Récompense totale', fontsize=12)
    ax1.set_title('V-Learning avec Goal Dynamique - Récompenses', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique du nombre de pas
    ax2.plot(episodes, steps_history, alpha=0.5, color='green', linewidth=0.5)
    
    if len(steps_history) >= window:
        steps_smooth = np.convolve(steps_history, 
                                   np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(steps_history)), steps_smooth, 
                color='orange', linewidth=2, label=f'Moyenne mobile ({window})')
    
    # Marquer les changements de goal
    for ep, goal_pos in goal_changes.items():
        ax2.axvline(x=ep, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax2.text(ep, ax2.get_ylim()[1]*0.9, f'Goal→{goal_pos}', 
                rotation=90, va='top', fontsize=9, color='green', fontweight='bold')
    
    ax2.set_xlabel('Épisode', fontsize=12)
    ax2.set_ylabel('Nombre de pas', fontsize=12)
    ax2.set_title('V-Learning avec Goal Dynamique - Nombre de pas', 
                 fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def train_with_changing_goal(env, agent, goal_schedule, n_episodes=2000, 
                             visualize_episodes=None):
    """
    Entraîne l'agent avec un goal qui change au cours des épisodes.
    
    Args:
        env: L'environnement
        agent: L'agent V-Learning
        goal_schedule (dict): {episode: nouvelle_position_goal}
        n_episodes (int): Nombre total d'épisodes
        visualize_episodes (list): Episodes à visualiser
    
    Returns:
        rewards_history, steps_history
    """
    rewards_history = []
    steps_history = []
    
    if visualize_episodes is None:
        visualize_episodes = []
    
    current_goal = env.goals[0]  # Goal initial
    
    for episode in range(n_episodes):
        # Changer le goal si nécessaire
        if episode in goal_schedule:
            new_goal = goal_schedule[episode]
            env.set_goal(new_goal)
            current_goal = new_goal
            print(f"\n🎯 Épisode {episode}: Goal déplacé vers {new_goal}")
        
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 200
        
        # Si on doit visualiser cet épisode
        should_visualize = episode in visualize_episodes
        
        if should_visualize:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))
        
        for step in range(max_steps):
            # Visualiser l'état actuel
            if should_visualize:
                agent._visualize_step(env, state, ax, episode, step, total_reward)
            
            # Choisir et exécuter une action
            action = agent.choose_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Mettre à jour V(s)
            agent.update(state, reward, next_state, terminated or truncated)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if terminated or truncated:
                if should_visualize:
                    agent._visualize_step(env, state, ax, episode, step + 1, 
                                       total_reward, final=True)
                break
        
        if should_visualize:
            plt.ioff()
            plt.close(fig)
        
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        # Afficher les progrès
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(rewards_history[-200:])
            avg_steps = np.mean(steps_history[-200:])
            print(f"Épisode {episode + 1}/{n_episodes} - "
                  f"Goal actuel: {current_goal} - "
                  f"Récompense moyenne: {avg_reward:.2f} - "
                  f"Pas moyens: {avg_steps:.2f}")
    
    print(f"\n✓ Entraînement terminé après {n_episodes} épisodes")
    return rewards_history, steps_history


def main():
    """
    Fonction principale pour tester V-Learning avec goal dynamique.
    """
    print("="*70)
    print("V-LEARNING AVEC GOAL DYNAMIQUE")
    print("="*70)
    print()
    
    # Créer l'environnement
    env = GridWorldEnv(
        size=5,
        start=(0, 0),
        obstacles=[(1, 1), (2, 2)],
        goals=[(4, 4)]  # Goal initial
    )
    
    print(f"Environnement: Grille {env.size}x{env.size}")
    print(f"Départ: {env.start_pos}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Goal initial: {env.goals[0]}")
    print()
    
    # Créer l'agent
    agent = VLearningAgent(
        n_states=env.observation_space.n,
        epsilon=0.1,
        alpha=0.1,
        gamma=0.99
    )
    
    # Définir le planning des changements de goal
    goal_schedule = {
        0: (4, 4),      # Episode 0-499: Goal en bas à droite
        500: (0, 4),    # Episode 500-999: Goal en haut à droite
        1000: (4, 0),   # Episode 1000-1499: Goal en bas à gauche
        1500: (2, 3),   # Episode 1500-1999: Goal au milieu-droit
    }
    
    print("📅 Planning des changements de goal:")
    for ep, goal in goal_schedule.items():
        print(f"   - Épisode {ep}: Goal → {goal}")
    print()
    
    # Episodes à visualiser (un par phase)
    visualize_episodes = [50, 550, 1050, 1550]
    
    print("🎬 Episodes visualisés: 50, 550, 1050, 1550")
    print()
    print("Démarrage de l'entraînement...")
    print()
    
    # Entraîner avec goal changeant
    rewards_history, steps_history = train_with_changing_goal(
        env, 
        agent, 
        goal_schedule, 
        n_episodes=2000,
        visualize_episodes=visualize_episodes
    )
    
    # Visualisation des résultats
    print("\n=== Génération du graphique d'apprentissage ===")
    plot_learning_with_goal_changes(rewards_history, steps_history, goal_schedule)
    
    # Test final avec chaque goal
    print("\n=== Tests finaux ===")
    test_goals = [(4, 4), (0, 4), (4, 0), (2, 3)]
    
    for test_goal in test_goals:
        env.set_goal(test_goal)
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(100):
            action = agent.act(state, env)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Goal {test_goal}: {steps} pas, récompense {total_reward:.1f}")
    
    print("\n" + "="*70)
    print("EXPÉRIENCE TERMINÉE!")
    print("="*70)
    print("\nObservations:")
    print("- L'agent s'adapte à chaque nouveau goal")
    print("- Performance baisse temporairement à chaque changement")
    print("- L'agent réapprend rapidement grâce à sa V-table existante")


if __name__ == "__main__":
    main()
