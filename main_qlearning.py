"""
Script principal pour comparer les 3 algorithmes:
- Value Iteration (programmation dynamique)
- V-Learning (TD avec fonction V)
- Q-Learning (TD avec fonction Q)
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from optimal_agent import ValueIterationAgent
from v_learning_agent import VLearningAgent
from q_learning_agent import QLearningAgent


def plot_comparison_learning_curves(v_rewards, v_steps, q_rewards, q_steps):
    """
    Compare les courbes d'apprentissage de V-Learning et Q-Learning.
    
    Args:
        v_rewards: Récompenses V-Learning
        v_steps: Pas V-Learning
        q_rewards: Récompenses Q-Learning
        q_steps: Pas Q-Learning
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    window = 50
    
    # Courbes de récompenses
    if len(v_rewards) >= window:
        v_smooth = np.convolve(v_rewards, np.ones(window)/window, mode='valid')
        q_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
        
        ax1.plot(range(window-1, len(v_rewards)), v_smooth, 
                color='purple', linewidth=2, label='V-Learning')
        ax1.plot(range(window-1, len(q_rewards)), q_smooth, 
                color='orange', linewidth=2, label='Q-Learning')
    
    ax1.set_xlabel('Épisode', fontsize=12)
    ax1.set_ylabel('Récompense totale (moyenne mobile)', fontsize=12)
    ax1.set_title('Comparaison: Récompenses', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Courbes de pas
    if len(v_steps) >= window:
        v_smooth_steps = np.convolve(v_steps, np.ones(window)/window, mode='valid')
        q_smooth_steps = np.convolve(q_steps, np.ones(window)/window, mode='valid')
        
        ax2.plot(range(window-1, len(v_steps)), v_smooth_steps, 
                color='purple', linewidth=2, label='V-Learning')
        ax2.plot(range(window-1, len(q_steps)), q_smooth_steps, 
                color='orange', linewidth=2, label='Q-Learning')
    
    ax2.set_xlabel('Épisode', fontsize=12)
    ax2.set_ylabel('Nombre de pas (moyenne mobile)', fontsize=12)
    ax2.set_title('Comparaison: Nombre de pas', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_three_algorithms(env, v_table_vi, v_table_vl, v_table_ql):
    """
    Compare les tables de valeurs des 3 algorithmes.
    
    Args:
        env: L'environnement
        v_table_vi: Table V de Value Iteration
        v_table_vl: Table V de V-Learning
        v_table_ql: Table V extraite de Q-Learning
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    
    # Value Iteration
    V_grid1 = v_table_vi.reshape(env.size, env.size)
    im1 = ax1.imshow(V_grid1, cmap='viridis', interpolation='nearest')
    plt.colorbar(im1, ax=ax1, label='Valeur')
    for i in range(env.size):
        for j in range(env.size):
            ax1.text(j, i, f'{V_grid1[i, j]:.1f}', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=9)
    ax1.set_title('Value Iteration\n(Programmation Dynamique - Optimal)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('Colonne')
    ax1.set_ylabel('Ligne')
    
    # V-Learning
    V_grid2 = v_table_vl.reshape(env.size, env.size)
    im2 = ax2.imshow(V_grid2, cmap='viridis', interpolation='nearest')
    plt.colorbar(im2, ax=ax2, label='Valeur')
    for i in range(env.size):
        for j in range(env.size):
            ax2.text(j, i, f'{V_grid2[i, j]:.1f}', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=9)
    ax2.set_title('V-Learning\n(TD avec V(s))', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Colonne')
    ax2.set_ylabel('Ligne')
    
    # Q-Learning
    V_grid3 = v_table_ql.reshape(env.size, env.size)
    im3 = ax3.imshow(V_grid3, cmap='viridis', interpolation='nearest')
    plt.colorbar(im3, ax=ax3, label='Valeur')
    for i in range(env.size):
        for j in range(env.size):
            ax3.text(j, i, f'{V_grid3[i, j]:.1f}', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=9)
    ax3.set_title('Q-Learning\n(TD avec Q(s,a))', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Colonne')
    ax3.set_ylabel('Ligne')
    
    # Différences moyennes
    diff_vl = np.abs(V_grid1 - V_grid2)
    diff_ql = np.abs(V_grid1 - V_grid3)
    diff_vl_ql = np.abs(V_grid2 - V_grid3)
    
    ax4.axis('off')
    stats_text = f"""
    STATISTIQUES DE COMPARAISON
    
    Erreur moyenne VS Value Iteration:
    • V-Learning: {np.mean(diff_vl):.3f}
    • Q-Learning: {np.mean(diff_ql):.3f}
    
    Erreur max VS Value Iteration:
    • V-Learning: {np.max(diff_vl):.3f}
    • Q-Learning: {np.max(diff_ql):.3f}
    
    Différence V-Learning vs Q-Learning:
    • Moyenne: {np.mean(diff_vl_ql):.3f}
    • Maximum: {np.max(diff_vl_ql):.3f}
    
    CONVERGENCE:
    ✓ Value Iteration: Exact (optimal)
    ✓ V-Learning: Approximation par expérience
    ✓ Q-Learning: Approximation par expérience
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    ax4.set_title('Analyse comparative', fontsize=12, fontweight='bold')
    
    # Marquer obstacles et goals
    for ax in [ax1, ax2, ax3]:
        for obs in env.obstacles:
            ax.scatter(obs[1], obs[0], marker='X', s=300, c='red', linewidths=2)
        for goal in env.goals:
            ax.scatter(goal[1], goal[0], marker='*', s=300, c='gold', linewidths=2)
    
    plt.tight_layout()
    plt.show()


def visualize_q_table(env, q_agent):
    """
    Visualise la Q-table avec des flèches indiquant la meilleure action.
    
    Args:
        env: L'environnement
        q_agent: L'agent Q-Learning
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Créer la grille avec les valeurs V (max Q)
    V = q_agent.get_V_from_Q()
    V_grid = V.reshape(env.size, env.size)
    
    im = ax.imshow(V_grid, cmap='viridis', interpolation='nearest', alpha=0.6)
    plt.colorbar(im, ax=ax, label='V(s) = max Q(s,a)')
    
    # Dessiner les flèches pour chaque état
    action_symbols = ['↑', '↓', '←', '→']
    action_colors = ['red', 'blue', 'green', 'purple']
    
    for i in range(env.size):
        for j in range(env.size):
            state = env._pos_to_state((i, j))
            
            # Ne pas dessiner sur obstacles ou goals
            if (i, j) in env.obstacles or (i, j) in env.goals:
                continue
            
            # Meilleure action
            best_action = np.argmax(q_agent.Q[state])
            
            # Dessiner la flèche
            ax.text(j, i, action_symbols[best_action], ha='center', va='center',
                   fontsize=25, color=action_colors[best_action], 
                   fontweight='bold', zorder=10)
            
            # Valeur
            ax.text(j, i+0.35, f'{V_grid[i, j]:.1f}', ha='center', va='center',
                   fontsize=8, color='white', fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    # Obstacles et goals
    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8,
                                   fill=True, color='gray', alpha=0.8))
        ax.text(obs[1], obs[0], 'X', ha='center', va='center',
               fontsize=20, fontweight='bold', color='red')
    
    for goal in env.goals:
        ax.add_patch(plt.Circle((goal[1], goal[0]), 0.35,
                               fill=True, color='gold', alpha=0.9))
        ax.text(goal[1], goal[0], '★', ha='center', va='center',
               fontsize=25, color='darkgreen')
    
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xlabel('Colonne', fontsize=12)
    ax.set_ylabel('Ligne', fontsize=12)
    ax.set_title('Q-Table: Politique optimale (flèches) et Valeurs V', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Fonction principale pour comparer les 3 algorithmes.
    """
    print("="*70)
    print("COMPARAISON: VALUE ITERATION vs V-LEARNING vs Q-LEARNING")
    print("="*70)
    print()
    
    # Créer l'environnement
    env = GridWorldEnv(
        size=5,
        start=(0, 0),
        obstacles=[(1, 1), (2, 2)],
        goals=[(4, 4)]
    )
    
    print(f"Environnement: Grille {env.size}x{env.size}")
    print(f"Départ: {env.start_pos}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Goals: {env.goals}")
    print()
    
    # 1. Value Iteration
    print("="*70)
    print("1. VALUE ITERATION (Programmation Dynamique)")
    print("="*70)
    vi_agent = ValueIterationAgent()
    V_table_vi, deltas = vi_agent.train(env, gamma=0.99, theta=1e-6)
    print()
    
    # 2. V-Learning
    print("="*70)
    print("2. V-LEARNING (Temporal Difference avec V)")
    print("="*70)
    print("Visualisation: épisodes 1, 250, 500, 1000\n")
    
    vl_agent = VLearningAgent(
        n_states=env.observation_space.n,
        epsilon=0.1,
        alpha=0.1,
        gamma=0.99
    )
    
    v_rewards, v_steps = vl_agent.train(
        env,
        n_episodes=1000,
        verbose=True,
        visualize_episodes=[1, 250, 500, 1000]
    )
    print()
    
    # 3. Q-Learning
    print("="*70)
    print("3. Q-LEARNING (Temporal Difference avec Q)")
    print("="*70)
    print("Visualisation: épisodes 1, 250, 500, 1000\n")
    
    ql_agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        epsilon=0.1,
        alpha=0.1,
        gamma=0.99
    )
    
    q_rewards, q_steps = ql_agent.train(
        env,
        n_episodes=1000,
        verbose=True,
        visualize_episodes=[1, 250, 500, 1000]
    )
    print()
    
    # Tests finaux
    print("="*70)
    print("TESTS FINAUX")
    print("="*70)
    
    for name, agent in [("Value Iteration", vi_agent),
                        ("V-Learning", vl_agent),
                        ("Q-Learning", ql_agent)]:
        env.reset()
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
        
        print(f"{name:20s}: {steps} pas, récompense {total_reward:.1f}")
    
    print()
    
    # Visualisations
    print("="*70)
    print("VISUALISATIONS")
    print("="*70)
    
    print("\n1. Comparaison des courbes d'apprentissage...")
    plot_comparison_learning_curves(v_rewards, v_steps, q_rewards, q_steps)
    
    print("2. Comparaison des tables de valeurs...")
    V_table_ql = ql_agent.get_V_from_Q()
    compare_three_algorithms(env, V_table_vi, vl_agent.V, V_table_ql)
    
    print("3. Visualisation de la Q-Table avec politique...")
    visualize_q_table(env, ql_agent)
    
    print("\n" + "="*70)
    print("COMPARAISON TERMINÉE!")
    print("="*70)
    print("\nCONCLUSIONS:")
    print("• Value Iteration: Optimal, mais nécessite modèle de l'environnement")
    print("• V-Learning: Apprend V(s) par expérience, doit simuler actions")
    print("• Q-Learning: Apprend Q(s,a) par expérience, action directe !")
    print("• Q-Learning est le plus populaire en RL moderne (DQN, etc.)")


if __name__ == "__main__":
    main()
