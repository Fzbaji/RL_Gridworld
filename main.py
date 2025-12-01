"""
Script principal pour exécuter les agents et visualiser les résultats.
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from random_agent import RandomAgent
from optimal_agent import ValueIterationAgent
from v_learning_agent import VLearningAgent


def run_random_agent(env, max_steps=100):
    """
    Exécute un épisode avec l'agent aléatoire.
    
    Args:
        env: L'environnement
        max_steps (int): Nombre maximum de pas
    """
    agent = RandomAgent()
    observation, _ = env.reset()
    
    print("=== Exécution de l'Agent Random ===")
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.act(observation, env)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Épisode terminé en {step + 1} pas avec récompense totale: {total_reward}")
            break
    else:
        print(f"Épisode non terminé après {max_steps} pas. Récompense totale: {total_reward}")
    
    print("Agent Random terminé\n")


def plot_learning_curves(rewards_history, steps_history):
    """
    Affiche les courbes d'apprentissage (récompenses et pas).
    
    Args:
        rewards_history (list): Historique des récompenses
        steps_history (list): Historique du nombre de pas
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculer les moyennes mobiles
    window = 50
    if len(rewards_history) >= window:
        rewards_smooth = np.convolve(rewards_history, 
                                     np.ones(window)/window, mode='valid')
        steps_smooth = np.convolve(steps_history, 
                                   np.ones(window)/window, mode='valid')
    else:
        rewards_smooth = rewards_history
        steps_smooth = steps_history
    
    # Graphique des récompenses
    ax1.plot(rewards_history, alpha=0.3, color='blue', label='Récompenses brutes')
    ax1.plot(range(len(rewards_smooth)), rewards_smooth, 
             color='red', linewidth=2, label=f'Moyenne mobile ({window})')
    ax1.set_xlabel('Épisode', fontsize=12)
    ax1.set_ylabel('Récompense totale', fontsize=12)
    ax1.set_title('Courbe d\'apprentissage - Récompenses', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique du nombre de pas
    ax2.plot(steps_history, alpha=0.3, color='green', label='Pas bruts')
    ax2.plot(range(len(steps_smooth)), steps_smooth, 
             color='orange', linewidth=2, label=f'Moyenne mobile ({window})')
    ax2.set_xlabel('Épisode', fontsize=12)
    ax2.set_ylabel('Nombre de pas', fontsize=12)
    ax2.set_title('Courbe d\'apprentissage - Nombre de pas', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_value_tables(env, v_iter_table, v_learning_table):
    """
    Compare les tables de valeurs de Value Iteration et V-Learning.
    
    Args:
        env: L'environnement
        v_iter_table: Table V de Value Iteration
        v_learning_table: Table V de V-Learning
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Value Iteration
    V_grid1 = v_iter_table.reshape(env.size, env.size)
    im1 = ax1.imshow(V_grid1, cmap='viridis', interpolation='nearest')
    plt.colorbar(im1, ax=ax1, label='Valeur')
    for i in range(env.size):
        for j in range(env.size):
            ax1.text(j, i, f'{V_grid1[i, j]:.1f}', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=10)
    ax1.set_title('Value Iteration\n(Programmation Dynamique)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Colonne')
    ax1.set_ylabel('Ligne')
    
    # V-Learning
    V_grid2 = v_learning_table.reshape(env.size, env.size)
    im2 = ax2.imshow(V_grid2, cmap='viridis', interpolation='nearest')
    plt.colorbar(im2, ax=ax2, label='Valeur')
    for i in range(env.size):
        for j in range(env.size):
            ax2.text(j, i, f'{V_grid2[i, j]:.1f}', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=10)
    ax2.set_title('V-Learning\n(Temporal Difference)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Colonne')
    ax2.set_ylabel('Ligne')
    
    # Différence
    diff_grid = np.abs(V_grid1 - V_grid2)
    im3 = ax3.imshow(diff_grid, cmap='Reds', interpolation='nearest')
    plt.colorbar(im3, ax=ax3, label='Différence absolue')
    for i in range(env.size):
        for j in range(env.size):
            ax3.text(j, i, f'{diff_grid[i, j]:.1f}', ha='center', va='center', 
                    color='black', fontweight='bold', fontsize=10)
    ax3.set_title('Différence absolue\n|VI - VL|', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Colonne')
    ax3.set_ylabel('Ligne')
    
    # Marquer obstacles et goals sur tous les graphiques
    for ax in [ax1, ax2, ax3]:
        for obs in env.obstacles:
            ax.scatter(obs[1], obs[0], marker='X', s=300, c='red', linewidths=2)
        for goal in env.goals:
            ax.scatter(goal[1], goal[0], marker='*', s=300, c='gold', linewidths=2)
    
    plt.tight_layout()
    plt.show()


def plot_convergence(deltas):
    """
    Affiche le graphique de convergence (Delta vs Itérations).
    
    Args:
        deltas (list): Historique des deltas
    """
    plt.figure(figsize=(10, 6))
    plt.plot(deltas, linewidth=2)
    plt.xlabel('Itération', fontsize=12)
    plt.ylabel('Delta (Erreur)', fontsize=12)
    plt.title('Convergence de Value Iteration', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_value_heatmap(env, V_table):
    """
    Affiche une heatmap des valeurs des états.
    
    Args:
        env: L'environnement
        V_table (np.array): Table des valeurs
    """
    # Convertir V_table en grille 2D
    V_grid = V_table.reshape(env.size, env.size)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(V_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Valeur de l\'état')
    
    # Ajouter les valeurs dans chaque cellule
    for i in range(env.size):
        for j in range(env.size):
            value = V_grid[i, j]
            plt.text(j, i, f'{value:.1f}', ha='center', va='center', 
                    color='white', fontweight='bold')
    
    # Marquer les obstacles et goals
    for obs in env.obstacles:
        plt.scatter(obs[1], obs[0], marker='X', s=500, c='red', linewidths=2)
    
    for goal in env.goals:
        plt.scatter(goal[1], goal[0], marker='*', s=500, c='gold', linewidths=2)
    
    plt.title('Heatmap des Valeurs des États (V-table)', fontsize=14, fontweight='bold')
    plt.xlabel('Colonne', fontsize=12)
    plt.ylabel('Ligne', fontsize=12)
    plt.xticks(range(env.size))
    plt.yticks(range(env.size))
    plt.tight_layout()
    plt.show()


def animate_agent_movement(env, agent, delay=0.5):
    """
    Anime le déplacement de l'agent dans le GridWorld.
    
    Args:
        env: L'environnement
        agent: L'agent entraîné
        delay (float): Délai entre chaque mouvement en secondes
    """
    import time
    
    observation, _ = env.reset()
    
    # Configuration de la figure
    plt.ion()  # Mode interactif
    fig, ax = plt.subplots(figsize=(8, 8))
    
    max_steps = env.size * env.size * 2
    step_count = 0
    
    for step in range(max_steps):
        ax.clear()
        
        # Créer la grille de base
        grid = np.ones((env.size, env.size))
        
        # Marquer les obstacles en gris foncé
        for obs in env.obstacles:
            grid[obs] = 0.3
        
        # Marquer les goals en doré
        for goal in env.goals:
            grid[goal] = 0.7
        
        # Afficher la grille
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
        current_pos = env._state_to_pos(observation)
        ax.add_patch(plt.Circle((current_pos[1], current_pos[0]), 0.3, 
                               fill=True, color='blue', alpha=0.9, zorder=10))
        ax.text(current_pos[1], current_pos[0], '●', ha='center', va='center', 
               fontsize=30, color='white', fontweight='bold')
        
        # Marquer le départ
        if step == 0:
            ax.scatter(env.start_pos[1], env.start_pos[0], marker='s', 
                      s=500, c='lightgreen', alpha=0.5, zorder=1, label='Départ')
        
        # Ajouter la grille
        ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        
        # Ajouter les labels
        ax.set_xlabel('Colonne', fontsize=12)
        ax.set_ylabel('Ligne', fontsize=12)
        ax.set_title(f'Agent se déplace dans le GridWorld - Pas: {step}', 
                    fontsize=14, fontweight='bold')
        
        # Légende
        action_names = ['↑ Haut', '↓ Bas', '← Gauche', '→ Droite']
        if step > 0:
            ax.text(0.02, 0.98, f'Dernière action: {action_names[action]}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(delay)
        
        # Exécuter l'action
        action = agent.act(observation, env)
        observation, reward, terminated, truncated, _ = env.step(action)
        step_count = step + 1
        
        if terminated or truncated:
            # Afficher le résultat final
            ax.text(0.5, 0.02, '✓ GOAL ATTEINT!', transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', color='green',
                   ha='center', verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            plt.draw()
            plt.pause(2)
            break
    
    plt.ioff()
    plt.show()
    
    print(f"\n✓ Agent a atteint le goal en {step_count} pas!")
    return step_count


def main():
    """
    Fonction principale pour exécuter le projet.
    """
    print("="*60)
    print("PROJET REINFORCEMENT LEARNING - GRIDWORLD")
    print("="*60)
    print()
    
    # Créer l'environnement avec 1 goal et 2 obstacles
    env = GridWorldEnv(
        size=5,
        start=(0, 0),
        obstacles=[(1, 1), (2, 2)],
        goals=[(4, 4)]
    )
    
    print(f"Environnement créé: Grille {env.size}x{env.size}")
    print(f"Départ: {env.start_pos}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Goals: {env.goals}")
    print()
    
    # Étape 1: Agent Random
    run_random_agent(env, max_steps=100)
    
    # Étape 2: Value Iteration Agent
    print("=== Entraînement de Value Iteration Agent ===")
    optimal_agent = ValueIterationAgent()
    V_table_vi, deltas = optimal_agent.train(env, gamma=0.99, theta=1e-6)
    print("Entraînement terminé\n")
    
    # Tester l'agent Value Iteration
    print("=== Test de l'Agent Value Iteration ===")
    observation, _ = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < 100:
        action = optimal_agent.act(observation, env)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            print(f"Épisode terminé en {steps} pas avec récompense totale: {total_reward}")
            break
    
    print()
    
    # Étape 3: V-Learning Agent
    print("=== Entraînement de V-Learning Agent (TD Learning) ===")
    print("L'agent va apprendre en jouant des épisodes...")
    print("Visualisation des épisodes: 1, 100, 500, 1000\n")
    
    v_learning_agent = VLearningAgent(
        n_states=env.observation_space.n,
        epsilon=0.1,
        alpha=0.1,
        gamma=0.99
    )
    
    rewards_history, steps_history = v_learning_agent.train(
        env, 
        n_episodes=1000, 
        verbose=True,
        visualize_episodes=[1, 100, 500, 1000]  # Visualiser ces épisodes
    )
    
    # Tester l'agent V-Learning
    print("\n=== Test de l'Agent V-Learning ===")
    observation, _ = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < 100:
        action = v_learning_agent.act(observation, env)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            print(f"Épisode terminé en {steps} pas avec récompense totale: {total_reward}")
            break
    
    print()
    
    # Étape 4: Visualisations
    print("=== Génération des Visualisations ===")
    
    # 1. Convergence Value Iteration
    print("1. Convergence de Value Iteration...")
    plot_convergence(deltas)
    
    # 2. Courbes d'apprentissage V-Learning
    print("2. Courbes d'apprentissage de V-Learning...")
    plot_learning_curves(rewards_history, steps_history)
    
    # 3. Comparaison des tables de valeurs
    print("3. Comparaison des tables de valeurs...")
    compare_value_tables(env, V_table_vi, v_learning_agent.V)
    
    # 4. Heatmap Value Iteration
    print("4. Heatmap Value Iteration...")
    plot_value_heatmap(env, V_table_vi)
    
    # 5. Heatmap V-Learning
    print("5. Heatmap V-Learning...")
    plot_value_heatmap(env, v_learning_agent.V)
    
    # 6. Animation Value Iteration
    print("6. Animation Value Iteration (après convergence)...")
    env_test = GridWorldEnv(size=5, start=(0, 0), 
                            obstacles=[(1, 1), (2, 2)], goals=[(4, 4)])
    animate_agent_movement(env_test, optimal_agent, delay=0.8)
    
    # Note: Animation V-Learning déjà montrée pendant l'entraînement
    
    print("\n" + "="*60)
    print("PROJET TERMINÉ AVEC SUCCÈS!")
    print("="*60)


if __name__ == "__main__":
    main()
