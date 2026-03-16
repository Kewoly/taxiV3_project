# DQN Taxi 15×10 — Deep Reinforcement Learning

Projet d'apprentissage par renforcement profond (DQN) sur un environnement Taxi custom 15×10 avec multi-passagers.

## 🎯 Objectif

Entraîner un agent DQN à livrer efficacement 2 passagers sur une grille 15×10, en **minimisant le nombre de pas** et en **maximisant les récompenses**.

## 📁 Structure du projet

```
.
├── src/
│   ├── envs/
│   │   ├── __init__.py
│   │   └── large_taxi.py      # Environnement custom 15×10
│   ├── agent.py               # Agent DQN (Stable Baselines 3)
│   └── interface.py           # Visualisations
├── training/
│   ├── train_dqn.py           # Script d'entraînement DQN
│   └── checkpoints/           # Modèles sauvegardés
├── notebooks/
│   ├── 01_training.ipynb      # Entraînement DQN
│   ├── 02_evaluation.ipynb    # Évaluation des performances
│   ├── 03_visualization.ipynb # Graphes pour présentation
│   └── 04_analysis.ipynb      # Analyse détaillée
├── logs/                      # TensorBoard + graphes
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📊 Utilisation

### Option 1 : Entraînement via script Python

```bash
python training/train_dqn.py --total_timesteps 300000 --learning_rate 3e-4
```

### Option 2 : Entraînement via Jupyter (recommandé)

```bash
jupyter notebook notebooks/01_training.ipynb
```

Puis exécuter les cellules pour lancer l'entraînement.

### Évaluation

```bash
jupyter notebook notebooks/02_evaluation.ipynb
```

### Visualisations pour présentation

```bash
jupyter notebook notebooks/03_visualization.ipynb
```

Génère des graphes intuitifs et colorés :
- Distribution des récompenses
- Distribution des pas (optimisation)
- Tableau de synthèse KPI
- Heatmap de couverture de l'espace
- Relation Récompense vs Pas

### Analyse détaillée

```bash
jupyter notebook notebooks/04_analysis.ipynb
```

Analyse des trajectoires, succès/échecs, visualisation de chemins.

## 🎮 Environnement

**Taxi 15×10 Multi-Passagers**

- **Grille** : 15 colonnes × 10 lignes
- **Passagers** : 2 passagers à ramasser et livrer
- **Actions** : 6 (Nord, Sud, Est, Ouest, Prendre, Déposer)
- **Observation** : Vecteur continu [taxi_x, taxi_y, p1_x, p1_y, p1_in_taxi, p1_dest_x, p1_dest_y, p2_x, p2_y, p2_in_taxi, p2_dest_x, p2_dest_y]
- **Récompenses** :
  - +20 : Livraison réussie d'un passager
  - +2 : Ramassage d'un passager
  - -1 : Chaque pas
  - Succès : Tous les passagers livrés

## 🧠 Agent DQN

**Hyperparamètres par défaut**

- **Learning rate** : 3e-4
- **Buffer size** : 100,000
- **Batch size** : 128
- **Gamma** : 0.99
- **Target update interval** : 1000
- **Exploration final eps** : 0.02
- **Exploration fraction** : 0.1
- **Total timesteps** : 300,000

## 📈 Résultats attendus

Après entraînement sur 300k timesteps :
- **Taux de succès** : 85-95%
- **Récompense moyenne** : 25-35
- **Pas moyens** : 40-60 (optimisé pour minimiser)
- **Couverture spatiale** : Exploration efficace de la grille

## 🎨 Palette de couleurs harmonisée

```python
COLORS = {
    'primary': '#2E86AB',      # Bleu
    'success': '#06A77D',      # Vert
    'warning': '#F77F00',      # Orange
    'danger': '#D62828',       # Rouge
    'light': '#EAE2B7',        # Beige
    'dark': '#1B1B1B',         # Noir
}
```

## 📊 TensorBoard

```bash
tensorboard --logdir logs/
```

Ouvrir `http://localhost:6006` pour visualiser :
- Reward par timestep
- Loss du réseau
- Exploration (epsilon)

## 📝 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_training.ipynb` | Entraînement complet du modèle DQN |
| `02_evaluation.ipynb` | Évaluation sur 100+ épisodes, métriques |
| `03_visualization.ipynb` | Graphes intuitifs pour présentation (5 graphes) |
| `04_analysis.ipynb` | Analyse détaillée des trajectoires |

## 🔍 Graphes de présentation

- **Distribution des récompenses** : Histogramme avec moyenne
- **Distribution des pas** : Optimisation du chemin
- **Tableau KPI** : Résumé synthétique des performances
- **Heatmap de visites** : Couverture de l'espace exploré
- **Reward vs Steps** : Relation efficacité/récompense

## ⚙️ Dépendances

- `gymnasium` : Environnement RL
- `stable-baselines3` : Implémentation DQN
- `torch` : Backend pour les réseaux de neurones
- `numpy`, `matplotlib`, `pandas` : Analyse et visualisation
- `jupyter` : Notebooks interactifs

## 📌 Notes

- Les modèles sont sauvegardés tous les 10k timesteps dans `training/checkpoints/`
- Les logs TensorBoard sont dans `logs/`
- Les graphes de présentation sont générés dans `logs/presentation_*.png`
- Tous les notebooks utilisent une palette de couleurs harmonisée
- L'optimisation des pas est un objectif clé (minimiser le nombre d'actions)
