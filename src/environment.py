"""
environment.py
--------------
Initialisation de l'environnement Gymnasium Taxi-v3 et formalisation du cadre RL.

Cadre RL — Taxi-v3 :
  State  : entier [0, 499] encodant (ligne taxi, col taxi, passager, destination)
  Action : 0=Sud 1=Nord 2=Est 3=Ouest 4=Prendre 5=Déposer
  Reward : +20 dépôt réussi | -10 prise/dépôt illégal | -1 chaque pas
"""

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
# Constantes décrivant le cadre RL
# ---------------------------------------------------------------------------

RL_FRAME = {
    "environment": "Taxi-v3",
    "state_space": "Discret — 500 états (5×5 grille × 5 positions passager × 4 destinations)",
    "action_space": "Discret — 6 actions (Sud, Nord, Est, Ouest, Prendre, Déposer)",
    "reward": {
        "depot_reussi": +20,
        "action_illegale": -10,
        "chaque_pas": -1,
    },
    "episode_terminé_si": "Dépôt réussi OU 200 pas dépassés",
}

ACTION_LABELS = {
    0: "Sud",
    1: "Nord",
    2: "Est",
    3: "Ouest",
    4: "Prendre",
    5: "Déposer",
}


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def make_env(render_mode: str = None) -> gym.Env:
    """
    Crée et retourne une instance de l'environnement Taxi-v3.

    Args:
        render_mode: None | "human" | "ansi" | "rgb_array"

    Returns:
        env: instance gymnasium.Env prête à l'emploi
    """
    env = gym.make("Taxi-v3", render_mode=render_mode)
    return env


def get_state_info(env: gym.Env, state: int) -> dict:
    """
    Décode un état entier en composantes lisibles.

    Args:
        env   : environnement Taxi-v3
        state : entier [0, 499]

    Returns:
        dict avec taxi_row, taxi_col, passenger_loc, destination
    """
    decoded = list(env.unwrapped.decode(state))
    return {
        "taxi_row":      decoded[0],
        "taxi_col":      decoded[1],
        "passenger_loc": decoded[2],  # 0=R 1=G 2=Y 3=B 4=dans le taxi
        "destination":   decoded[3],  # 0=R 1=G 2=Y 3=B
    }


def describe_env() -> None:
    """Affiche un résumé du cadre RL dans la console."""
    print("=" * 55)
    print("  Environnement :", RL_FRAME["environment"])
    print("  State space   :", RL_FRAME["state_space"])
    print("  Action space  :", RL_FRAME["action_space"])
    print("  Rewards       :")
    for k, v in RL_FRAME["reward"].items():
        print(f"    {k:<20} {v:+d}")
    print("  Fin épisode   :", RL_FRAME["episode_terminé_si"])
    print("=" * 55)
