# DQN Taxi 10×8 - Projet Simple et Fonctionnel

Architecture légère et simple pour entraîner un agent DQN sur l'environnement Taxi.

## Structure

```
taxiV3_simple/
├── env.py              # Environnement Taxi 10×8
├── train.py            # Script d'entraînement
├── evaluate.py         # Script d'évaluation
├── notebook.ipynb      # Workflow complet en notebook
├── requirements.txt    # Dépendances
└── README.md          # Ce fichier
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Option 1 : Notebook (recommandé)
```bash
jupyter notebook notebook.ipynb
```

Exécutez les cellules séquentiellement :
1. Test de l'environnement
2. Entraînement DQN (300k timesteps)
3. Évaluation (200 épisodes)
4. Visualisations

### Option 2 : Scripts Python

Entraîner :
```bash
python train.py
```

Évaluer :
```bash
python evaluate.py
```

## Environnement

- **Grille** : 10×8
- **Passagers** : 2
- **Actions** : 6 (Nord, Sud, Est, Ouest, Prendre, Déposer)
- **Récompenses** : Calibrées (-1 par pas, +5 pickup, +20 dropoff)
- **Observations** : Normalisées [0, 1]

## Hyperparamètres DQN (optimisés)

- Learning rate : 5e-4
- Buffer size : 50,000
- Batch size : 64
- Gamma : 0.99
- Target update interval : 500
- Exploration fraction : 0.2
- Total timesteps : 300,000

## Récompenses

- Mouvement : -1.0 par pas
- Pickup : +5.0
- Dropoff : +20.0

## Résultats attendus

- **Taux de succès** : > 80%
- **Récompense moyenne** : > 20
- **Pas moyens** : < 50

## Fichiers générés

- `models/dqn_taxi.zip` - Modèle entraîné
- `results.png` - Dashboard de visualisation
