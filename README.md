# 🎮 Projet Reinforcement Learning - GridWorld

## 📋 Table des matières
1. [Introduction](#introduction)
2. [Architecture du projet](#architecture-du-projet)
3. [Concepts théoriques](#concepts-théoriques)
4. [Environnement GridWorld](#environnement-gridworld)
5. [Agents implémentés](#agents-implémentés)
6. [Installation et utilisation](#installation-et-utilisation)
7. [Résultats et comparaisons](#résultats-et-comparaisons)
8. [Expérimentations avancées](#expérimentations-avancées)

---

## 🎯 Introduction

Ce projet est une implémentation complète de plusieurs algorithmes d'apprentissage par renforcement (Reinforcement Learning) appliqués à un environnement GridWorld. Il compare trois approches fondamentales :

- **Value Iteration** (Programmation Dynamique)
- **V-Learning** (Temporal Difference avec fonction V)
- **Q-Learning** (Temporal Difference avec fonction Q)

### Objectif pédagogique
Comprendre les différences entre les algorithmes model-based et model-free, et observer comment les agents apprennent à naviguer dans un environnement avec obstacles pour atteindre un objectif.

---

## 📁 Architecture du projet

```
RL_Gridworld/
│
├── grid_env.py              # Environnement GridWorld (hérite de gym.Env)
├── random_agent.py          # Agent aléatoire (baseline)
├── optimal_agent.py         # Value Iteration (DP)
├── v_learning_agent.py      # V-Learning (TD)
├── q_learning_agent.py      # Q-Learning (TD)
│
├── main.py                  # Script principal (Value Iteration + V-Learning)
├── main_qlearning.py        # Comparaison des 3 algorithmes
├── dynamic_goal_experiment.py  # Expérience avec goal dynamique
│
├── requirements.txt         # Dépendances Python
└── README.md               # Documentation (ce fichier)
```

---

## 📚 Concepts théoriques

### 1. Apprentissage par Renforcement (RL)

L'apprentissage par renforcement est un paradigme d'apprentissage automatique où un **agent** apprend à prendre des **actions** dans un **environnement** pour maximiser une **récompense cumulative**.

**Composants clés :**
- **État (State)** : Situation actuelle de l'agent
- **Action (Action)** : Choix que l'agent peut faire
- **Récompense (Reward)** : Signal de feedback de l'environnement
- **Politique (Policy)** : Stratégie de l'agent (quelle action choisir dans chaque état)
- **Fonction de valeur** : Estimation de la récompense future attendue

### 2. Processus de Décision Markovien (MDP)

Un MDP est défini par :
- **S** : Ensemble d'états
- **A** : Ensemble d'actions
- **P(s'|s,a)** : Probabilité de transition de l'état s vers s' en effectuant l'action a
- **R(s,a,s')** : Récompense reçue lors de la transition
- **γ (gamma)** : Facteur de discount (0 ≤ γ ≤ 1)

### 3. Équation de Bellman

L'équation fondamentale du RL qui exprime la valeur d'un état :

```
V(s) = max[R(s,a) + γ Σ P(s'|s,a) V(s')]
        a            s'
```

Pour Q-Learning :
```
Q(s,a) = R(s,a) + γ Σ P(s'|s,a) max Q(s',a')
                    s'           a'
```

---

## 🌍 Environnement GridWorld

### Description

`grid_env.py` - Classe **GridWorldEnv** héritant de `gym.Env` (Gymnasium)

**Caractéristiques :**
- Grille 2D de taille configurable (par défaut 5×5)
- Position de départ : (0,0)
- Obstacles : Positions infranchissables
- Goal : Objectif à atteindre
- Actions : 4 directions (Haut, Bas, Gauche, Droite)

### Espace d'observation
```python
observation_space = spaces.Discrete(size * size)
```
Chaque case de la grille correspond à un état unique (0 à 24 pour une grille 5×5).

### Espace d'action
```python
action_space = spaces.Discrete(4)
```
- 0 : Haut (↑)
- 1 : Bas (↓)
- 2 : Gauche (←)
- 3 : Droite (→)

### Fonction de récompense
```python
reward = -1  # Pénalité pour chaque pas (encourage les chemins courts)
reward = 10  # Récompense pour atteindre le goal
```

### Dynamique de transition
- Si l'agent tape un **mur** ou un **obstacle**, il reste sur place
- Sinon, il se déplace dans la direction choisie
- L'épisode se termine quand le goal est atteint (`terminated = True`)

### Méthodes principales

```python
reset() → observation, info
step(action) → observation, reward, terminated, truncated, info
set_goal(new_goal) → None  # Change la position du goal dynamiquement
```

---

## 🤖 Agents implémentés

### 1. Random Agent (Agent Aléatoire)

**Fichier :** `random_agent.py`

**Principe :**
- Sélectionne une action aléatoire à chaque pas
- Sert de **baseline** pour comparer les performances

**Code :**
```python
def act(self, observation, env):
    return env.action_space.sample()
```

**Performance :** Très faible, met beaucoup de temps (voire ne termine jamais)

---

### 2. Value Iteration Agent

**Fichier :** `optimal_agent.py`

#### Principe théorique

**Type :** Programmation Dynamique (Model-Based)

**Ce qu'il apprend :** Fonction de valeur **V(s)** pour chaque état

**Formule de mise à jour :**
```
V(s) ← max[R(s,a) + γ V(s')]
        a
```

**Algorithme :**
1. Initialiser V(s) = 0 pour tous les états
2. **Itérer jusqu'à convergence :**
   - Pour chaque état s :
     - Pour chaque action a :
       - Calculer la valeur : R + γ V(s')
     - V(s) = max de ces valeurs
   - Calculer delta = max|V_ancien - V_nouveau|
   - Si delta < θ (seuil), arrêter
3. Extraire la politique : π(s) = argmax[R + γ V(s')]

#### Caractéristiques

✅ **Avantages :**
- Trouve la solution **optimale**
- Converge rapidement (9 itérations dans notre cas)
- Pas besoin d'exploration

❌ **Inconvénients :**
- Nécessite un **modèle complet** de l'environnement (transitions, récompenses)
- Doit simuler toutes les actions pour choisir
- Non applicable aux environnements inconnus

#### Features apprises
**V(état)** → "Quelle est la valeur d'être dans cet état ?"

Exemple : V[(2,3)] = 5.8 signifie "être en position (2,3) vaut 5.8"

---

### 3. V-Learning Agent (TD Learning)

**Fichier :** `v_learning_agent.py`

#### Principe théorique

**Type :** Temporal Difference Learning (Model-Free)

**Ce qu'il apprend :** Fonction de valeur **V(s)** par expérience

**Formule de mise à jour :**
```
V(s) ← V(s) + α[R + γ V(s') - V(s)]
                 └─────┬─────┘
                   TD target
```

Où :
- **α (alpha)** : Taux d'apprentissage (0.1)
- **γ (gamma)** : Facteur de discount (0.99)
- **TD target** : Estimation basée sur l'expérience réelle

**Algorithme :**
1. Initialiser V(s) = 0 pour tous les états
2. **Pour chaque épisode :**
   - Partir de l'état initial
   - **Pour chaque pas :**
     - Choisir action avec ε-greedy (exploration/exploitation)
     - Exécuter l'action, observer R et s'
     - Mettre à jour : V(s) ← V(s) + α[R + γV(s') - V(s)]
     - s ← s'
   - Terminer si goal atteint

#### Politique ε-greedy

```python
if random() < ε:
    action = random()  # Exploration (10%)
else:
    action = argmax Q_estimé  # Exploitation (90%)
```

#### Caractéristiques

✅ **Avantages :**
- **Model-free** : Apprend par interaction
- S'adapte aux environnements inconnus
- Apprentissage en ligne

❌ **Inconvénients :**
- Converge plus lentement que Value Iteration
- Nécessite beaucoup d'épisodes (1000+)
- Doit **simuler** toutes les actions pour choisir (comme Value Iteration)

#### Features apprises
**V(état)** → Même que Value Iteration, mais appris par expérience

---

### 4. Q-Learning Agent

**Fichier :** `q_learning_agent.py`

#### Principe théorique

**Type :** Temporal Difference Learning (Model-Free)

**Ce qu'il apprend :** Fonction Q-valeur **Q(s, a)** pour chaque paire état-action

**Formule de mise à jour :**
```
Q(s,a) ← Q(s,a) + α[R + γ max Q(s',a') - Q(s,a)]
                           a'
```

**Différence clé avec V-Learning :**
- V-Learning : V(état) → 25 valeurs (grille 5×5)
- Q-Learning : Q(état, action) → 25 × 4 = **100 valeurs**

**Algorithme :**
1. Initialiser Q(s,a) = 0 pour tous les états et actions
2. **Pour chaque épisode :**
   - Partir de l'état initial
   - **Pour chaque pas :**
     - Choisir action avec ε-greedy
     - Exécuter l'action, observer R et s'
     - Mettre à jour : Q(s,a) ← Q(s,a) + α[R + γ max Q(s',a') - Q(s,a)]
     - s ← s'
   - Terminer si goal atteint

#### Caractéristiques

✅ **Avantages :**
- **Model-free** : Apprend par interaction
- **Action directe** : argmax Q(s,a) sans simulation !
- Off-policy : Peut apprendre d'expériences explorées différemment
- Base de Deep Q-Learning (DQN)

❌ **Inconvénients :**
- Table Q plus grande (mémoire)
- Converge plus lentement au début
- Nécessite beaucoup d'épisodes

#### Features apprises
**Q(état, action)** → "Quelle est la valeur de faire l'action a dans l'état s ?"

Exemples :
- Q[(2,3), Haut] = 6.5
- Q[(2,3), Bas] = 2.1
- Q[(2,3), Gauche] = 4.0
- Q[(2,3), Droite] = 7.2 ← **Meilleure action !**

#### Extraction de V depuis Q
```python
V(s) = max Q(s,a)
       a
```

---

## 🚀 Installation et utilisation

### Prérequis

- Python 3.8+
- pip

### Installation

1. **Cloner ou télécharger le projet**

2. **Créer un environnement virtuel (recommandé)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Installer les dépendances**
```powershell
pip install -r requirements.txt
```

**Dépendances :**
- gymnasium==1.2.2
- numpy==2.3.5
- matplotlib==3.10.7

### Exécution

#### Script 1 : Value Iteration + V-Learning
```powershell
python main.py
```

**Ce script exécute :**
1. Agent Random (baseline)
2. Value Iteration (convergence en 9 itérations)
3. V-Learning (1000 épisodes avec visualisations aux épisodes 1, 100, 500, 1000)

**Visualisations générées :**
- Convergence de Value Iteration
- Courbes d'apprentissage V-Learning
- Comparaison des tables V
- Heatmaps des valeurs
- Animation de l'agent se déplaçant

#### Script 2 : Comparaison des 3 algorithmes
```powershell
python main_qlearning.py
```

**Ce script compare :**
1. Value Iteration
2. V-Learning (1000 épisodes)
3. Q-Learning (1000 épisodes)

**Visualisations générées :**
- Courbes d'apprentissage comparées (V-Learning vs Q-Learning)
- Comparaison des 3 tables de valeurs
- Q-Table avec flèches de politique optimale
- Statistiques de performance

#### Script 3 : Expérience Goal Dynamique
```powershell
python dynamic_goal_experiment.py
```

**Ce script teste :**
- V-Learning avec goal qui change de position
- 2000 épisodes avec 4 changements de goal
- Visualisation de l'adaptation de l'agent

**Planning des changements :**
- Épisodes 0-499 : Goal à (4,4)
- Épisodes 500-999 : Goal à (0,4)
- Épisodes 1000-1499 : Goal à (4,0)
- Épisodes 1500-1999 : Goal à (2,3)

---

## 📊 Résultats et comparaisons

### Performance finale (Goal fixe à (4,4))

| Algorithme | Convergence | Pas optimal | Récompense | Modèle requis |
|------------|-------------|-------------|------------|---------------|
| **Value Iteration** | 9 itérations | 8 | +3.0 | ✅ Oui |
| **V-Learning** | 1000 épisodes | 8 | +3.0 | ❌ Non |
| **Q-Learning** | 1000 épisodes | 8 | +3.0 | ❌ Non |

### Vitesse de convergence

**Value Iteration :**
- ⚡ Très rapide (9 itérations)
- Calcul direct de la solution optimale

**V-Learning :**
- 🐢 Apprentissage progressif
- Récompense moyenne ~2.0 dès le début
- Stable tout au long de l'entraînement

**Q-Learning :**
- 🐌 Démarrage lent (récompense -5.42 aux 100 premiers épisodes)
- Rattrape et stabilise autour de ~2.2
- Converge vers la politique optimale

### Taille des tables

**Grille 5×5 :**
- Value Iteration : 25 valeurs (V-table)
- V-Learning : 25 valeurs (V-table)
- Q-Learning : **100 valeurs** (Q-table = 25 états × 4 actions)

### Avantages comparés

| Critère | Value Iteration | V-Learning | Q-Learning |
|---------|----------------|------------|------------|
| Optimalité | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Vitesse | ⭐⭐⭐ | ⭐ | ⭐ |
| Sans modèle | ❌ | ✅ | ✅ |
| Action directe | ❌ | ❌ | ✅ |
| Mémoire | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Scalabilité | ⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 🔬 Expérimentations avancées

### 1. Goal Dynamique

**Observation :** Quand le goal change de position, l'agent doit réapprendre.

**Pourquoi ?** L'agent apprend seulement **V(position)** ou **Q(position, action)**, mais ne sait pas **où est le goal**.

**Résultats :**
- À chaque changement de goal : **chute temporaire** de performance
- Puis **réapprentissage rapide** grâce à la V-table existante
- La connaissance des obstacles est conservée

**Graphique :** Les lignes vertes montrent les changements de goal et les perturbations associées.

### 2. Feature Engineering

**Problème actuel :**
- Feature = Position de l'agent uniquement
- L'agent ne "voit" pas le goal

**Amélioration possible :**
- Feature = (Position agent, Position goal)
- L'agent apprendrait : "Comment aller de A vers B" (généralisation)
- S'adapterait **instantanément** aux changements de goal

**Non implémenté dans ce projet** (reste simple pour la pédagogie).

### 3. Hyperparamètres

**Paramètres configurables :**

```python
epsilon = 0.1   # Taux d'exploration (10% actions aléatoires)
alpha = 0.1     # Taux d'apprentissage
gamma = 0.99    # Facteur de discount (importance du futur)
```

**Effets :**
- ↑ epsilon : Plus d'exploration, apprentissage plus lent mais robuste
- ↑ alpha : Apprentissage plus rapide mais moins stable
- ↑ gamma : Favorise les récompenses futures (chemins longs acceptables)

---

## 🎨 Visualisations

### 1. Heatmaps des valeurs

**Interprétation :**
- Couleurs chaudes (jaune) : États de haute valeur (proches du goal)
- Couleurs froides (violet) : États de faible valeur (loin du goal)
- Les valeurs diminuent en s'éloignant du goal

### 2. Courbes d'apprentissage

**Récompenses vs Épisodes :**
- Montre la progression de l'agent
- Moyenne mobile pour lisser le bruit

**Pas vs Épisodes :**
- Nombre de pas pour atteindre le goal
- Diminue au fur et à mesure de l'apprentissage

### 3. Q-Table avec flèches

**Visualisation unique à Q-Learning :**
- Flèches colorées indiquent la meilleure action dans chaque case
- Rouge (↑), Bleu (↓), Vert (←), Violet (→)
- Montre visuellement la **politique optimale**

### 4. Animations

**Agents se déplaçant :**
- Cercle bleu : Value Iteration
- Cercle violet : V-Learning
- Cercle orange : Q-Learning
- Affichage en temps réel du déplacement pas à pas

---

## 🧪 Notions clés apprises

### 1. Model-Based vs Model-Free

**Model-Based (Value Iteration) :**
- Connaît les transitions P(s'|s,a)
- Connaît les récompenses R(s,a)
- Calcule directement la solution optimale

**Model-Free (V-Learning, Q-Learning) :**
- Ne connaît PAS les transitions
- Apprend par **essai-erreur**
- S'adapte aux environnements inconnus

### 2. Exploration vs Exploitation

**Dilemme fondamental du RL :**
- **Exploration** : Essayer de nouvelles actions (découvrir)
- **Exploitation** : Utiliser les meilleures actions connues (optimiser)

**Solution ε-greedy :**
```python
if random() < ε:
    explore()  # 10%
else:
    exploit()  # 90%
```

### 3. On-Policy vs Off-Policy

**On-Policy (V-Learning dans notre implémentation) :**
- Apprend de la politique qu'il suit

**Off-Policy (Q-Learning) :**
- Peut apprendre d'expériences différentes
- Met à jour Q avec **max** même si l'action choisie était exploratoire

### 4. Temporal Difference (TD)

**Idée clé :** Mise à jour basée sur la différence entre :
- **Prédiction** : V(s) ou Q(s,a)
- **Cible** : R + γV(s') ou R + γ max Q(s',a')

**TD Error :**
```
δ = [R + γV(s')] - V(s)
```

C'est le "signal d'apprentissage".

---

## 📈 Extensions possibles

### 1. Deep Q-Learning (DQN)
Remplacer la Q-table par un **réseau de neurones** pour traiter des espaces d'états continus ou très larges.

### 2. Policy Gradient
Apprendre directement la politique π(a|s) au lieu de passer par Q ou V.

### 3. Actor-Critic
Combiner V-Learning (Critic) et Policy Gradient (Actor).

### 4. Multi-Agent RL
Plusieurs agents qui apprennent simultanément dans le même environnement.

### 5. Environnements plus complexes
- Stochasticité (transitions probabilistes)
- Récompenses partielles
- Observations partielles (POMDP)

---

## 🎓 Conclusion

Ce projet illustre les fondamentaux de l'apprentissage par renforcement :

### Ce qu'on a appris :

1. **Environnement GridWorld** : Création d'un MDP avec Gymnasium
2. **Value Iteration** : Programmation dynamique pour trouver la solution optimale
3. **V-Learning** : Apprentissage de V(s) par Temporal Difference
4. **Q-Learning** : Apprentissage de Q(s,a) pour action directe
5. **Visualisations** : Comprendre visuellement comment l'agent apprend
6. **Comparaisons** : Model-based vs model-free, vitesse vs flexibilité

### Points clés :

- **Value Iteration** : Optimal mais nécessite un modèle
- **Q-Learning** : Plus lent mais généralise mieux (base de DQN)
- **Features** : L'agent apprend seulement ce qu'il observe
- **Exploration** : Essentielle pour découvrir de nouvelles stratégies

### Applications réelles :

- Jeux (Atari, Go, Chess)
- Robotique (navigation, manipulation)
- Finance (trading algorithmique)
- Ressources (optimisation énergétique)
- Publicité (recommandation personnalisée)

---

## 📚 Références

- **Sutton & Barto** - Reinforcement Learning: An Introduction (2018)
- **Gymnasium Documentation** - https://gymnasium.farama.org/
- **OpenAI Spinning Up** - https://spinningup.openai.com/

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre d'un cours d'apprentissage par renforcement.

**Date :** Décembre 2025

---

## 📝 Licence

Ce projet est à usage éducatif.

---

**Bon apprentissage ! 🚀**
