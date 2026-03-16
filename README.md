# Taxi-v3 RL — Q-Learning Agent

Mini-projet Deep Reinforcement Learning — Bootcamp IA Ynov Campus  
Environnement : **Gymnasium Taxi-v3** | Algorithme : **Q-Learning**

---

## Cadre RL

| Élément | Description |
|---------|-------------|
| **Environnement** | Grille 5×5, taxi doit prendre un passager et le déposer à la bonne destination |
| **State** | Entier [0–499] encodant (ligne taxi, colonne taxi, position passager, destination) |
| **Action** | 6 actions discrètes : Sud, Nord, Est, Ouest, Prendre, Déposer |
| **Reward** | +20 dépôt réussi · −10 action illégale · −1 chaque pas |
| **Fin épisode** | Dépôt réussi OU 200 pas dépassés |

---

## Structure du projet

```
Taxi_RL_Project/
├── src/
│   ├── __init__.py
│   ├── environment.py   # Initialisation Gymnasium + cadre RL
│   ├── agent.py         # QLearningAgent + DQNAgent (SB3)
│   └── interface.py     # Render, animations, graphiques
├── training/
│   ├── train.py         # Boucle d'entraînement principale
│   └── checkpoints/     # Q-tables sauvegardées (.pkl)
├── evaluation/
│   └── evaluate.py      # Métriques finales + distribution
├── logs/                # TensorBoard + PNG des courbes
├── main.py              # Point d'entrée unique
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Entraînement

```bash
# Entraînement avec les paramètres par défaut (2000 épisodes)
python main.py train

# Personnaliser les hyperparamètres
python main.py train --episodes 3000 --alpha 0.15 --gamma 0.99
```

### 2. Visualiser les courbes (TensorBoard)

```bash
tensorboard --logdir logs
# puis ouvrir http://localhost:6006
```

### 3. Évaluation

```bash
# Évaluation sur 100 épisodes
python main.py eval

# Évaluation d'un checkpoint intermédiaire
python main.py eval --model training/checkpoints/qtable_ep1000.pkl --episodes 200
```

### 4. Démonstration visuelle

```bash
# Démo dans le terminal (rendu ANSI)
python main.py demo

# Démo plus lente pour la présentation
python main.py demo --delay 0.5
```

---

## Hyperparamètres

| Paramètre | Valeur par défaut | Rôle |
|-----------|------------------|------|
| `alpha` (α) | 0.1 | Taux d'apprentissage |
| `gamma` (γ) | 0.99 | Facteur d'actualisation (importance du futur) |
| `epsilon` (ε) | 1.0 → 0.01 | Exploration → Exploitation |
| `epsilon_decay` | 0.995 | Vitesse de décroissance de l'exploration |
| `episodes` | 2000 | Nombre total d'épisodes d'entraînement |

---

## Résultats attendus

Après ~2000 épisodes d'entraînement :

- **Reward moyen** : entre +7 et +10 (contre −800 pour un agent aléatoire)
- **Taux de succès** : > 95 %
- **Pas moyens** : ~13 (chemin optimal court)

---

## Algorithme — Q-Learning

Mise à jour de la Q-table à chaque pas :

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

**Pourquoi Q-Learning pour Taxi-v3 ?**
- Espace d'état discret et petit (500 états) → table en mémoire
- Convergence garantie vers la politique optimale
- Rapide à entraîner (~30 secondes pour 2000 épisodes)
- Interprétable : on peut inspecter directement la Q-table

---

## Livrables pour la soutenance

- `training/checkpoints/qtable_final.pkl` — modèle entraîné
- `logs/training_dashboard.png` — courbes d'apprentissage
- `logs/eval_distribution.png` — distribution des performances
- Démo live : `python main.py demo --delay 0.4`
