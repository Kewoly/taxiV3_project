"""
interface.py
------------
Fonctions d'affichage, d'animation et de visualisation des résultats.

Contient :
  - render_episode()    : joue un épisode en mode "human" (terminal ANSI)
  - plot_rewards()      : courbe de récompenses lissée
  - plot_epsilon()      : décroissance epsilon
  - plot_steps()        : nombre de pas par épisode
  - plot_all()          : tableau de bord complet (3 sous-graphes)
  - print_q_table_top() : affiche les meilleures actions de la Q-table
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional


# ===========================================================================
# Affichage d'un épisode en console
# ===========================================================================

def render_episode(env, agent, max_steps: int = 200, delay: float = 0.3) -> dict:
    """
    Joue un épisode complet en affichant chaque pas dans le terminal.

    Args:
        env       : environnement gymnasium (render_mode="human" ou "ansi")
        agent     : agent avec méthode select_action(state) -> int
        max_steps : limite de pas
        delay     : pause en secondes entre chaque pas

    Returns:
        dict avec total_reward, steps, success
    """
    from src.environment import ACTION_LABELS

    state, _ = env.reset()
    total_reward = 0
    success = False

    print("\n" + "=" * 40)
    print("  DÉMONSTRATION — Agent Taxi-v3")
    print("=" * 40)

    for step in range(max_steps):
        env.render()
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        label = ACTION_LABELS.get(action, str(action))
        print(f"  Étape {step+1:3d} | Action : {label:<8} | Reward : {reward:+3d} | Total : {total_reward:+4d}")

        total_reward += reward
        state = next_state

        if terminated or truncated:
            if reward == 20:
                success = True
                print("\n  ✓ Passager déposé avec succès !")
            else:
                print("\n  ✗ Épisode terminé (échec ou timeout).")
            break

        time.sleep(delay)

    print(f"\n  Résultat : reward={total_reward} | steps={step+1} | succès={success}")
    print("=" * 40 + "\n")
    return {"total_reward": total_reward, "steps": step + 1, "success": success}


# ===========================================================================
# Courbes d'apprentissage
# ===========================================================================

def _smooth(values: List[float], window: int = 50) -> np.ndarray:
    """Lisse une liste de valeurs avec une moyenne glissante."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_rewards(
    rewards: List[float],
    title: str = "Récompenses par épisode",
    window: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """
    Trace la courbe de récompenses brutes et lissées.

    Args:
        rewards   : liste de récompenses totales par épisode
        title     : titre du graphe
        window    : fenêtre de lissage
        save_path : chemin de sauvegarde PNG (None = affichage uniquement)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    episodes = np.arange(1, len(rewards) + 1)

    ax.plot(episodes, rewards, alpha=0.25, color="#7F77DD", linewidth=0.8, label="Brut")

    smoothed = _smooth(rewards, window)
    offset = len(rewards) - len(smoothed)
    ax.plot(
        np.arange(offset + 1, len(rewards) + 1),
        smoothed,
        color="#534AB7",
        linewidth=2,
        label=f"Lissé (fenêtre={window})",
    )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense totale")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] Sauvegardé → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_epsilon(
    epsilons: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Trace la décroissance d'epsilon au fil des épisodes.

    Args:
        epsilons  : liste des valeurs epsilon à chaque épisode
        save_path : chemin de sauvegarde PNG
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(epsilons, color="#1D9E75", linewidth=1.5)
    ax.fill_between(range(len(epsilons)), epsilons, alpha=0.15, color="#1D9E75")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Décroissance de l'exploration (ε-greedy)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] Sauvegardé → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_steps(
    steps: List[int],
    window: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """
    Trace le nombre de pas par épisode (doit décroître avec l'apprentissage).

    Args:
        steps     : liste du nombre de pas par épisode
        window    : fenêtre de lissage
        save_path : chemin de sauvegarde PNG
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(steps, alpha=0.2, color="#D85A30", linewidth=0.8)

    smoothed = _smooth(steps, window)
    offset = len(steps) - len(smoothed)
    ax.plot(
        np.arange(offset, len(steps)),
        smoothed,
        color="#993C1D",
        linewidth=2,
        label=f"Lissé (fenêtre={window})",
    )

    ax.set_xlabel("Épisode")
    ax.set_ylabel("Nombre de pas")
    ax.set_title("Pas par épisode (↓ = agent plus efficace)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] Sauvegardé → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_all(
    rewards: List[float],
    epsilons: List[float],
    steps: List[int],
    window: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """
    Tableau de bord complet : récompenses + epsilon + pas (3 sous-graphes).

    Args:
        rewards   : récompenses par épisode
        epsilons  : valeurs epsilon par épisode
        steps     : pas par épisode
        window    : fenêtre de lissage
        save_path : chemin de sauvegarde PNG
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Tableau de bord — Entraînement Taxi-v3 (Q-Learning)", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    # --- Récompenses ---
    ax1 = fig.add_subplot(gs[0])
    eps = np.arange(1, len(rewards) + 1)
    ax1.plot(eps, rewards, alpha=0.2, color="#7F77DD", linewidth=0.7)
    sm = _smooth(rewards, window)
    ax1.plot(np.arange(len(rewards) - len(sm) + 1, len(rewards) + 1), sm, color="#534AB7", linewidth=2)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.set_ylabel("Récompense")
    ax1.set_title("Récompenses totales par épisode")
    ax1.grid(True, alpha=0.25)

    # --- Epsilon ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epsilons, color="#1D9E75", linewidth=1.5)
    ax2.fill_between(range(len(epsilons)), epsilons, alpha=0.12, color="#1D9E75")
    ax2.set_ylabel("Epsilon (ε)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Décroissance de l'exploration")
    ax2.grid(True, alpha=0.25)

    # --- Pas ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(steps, alpha=0.2, color="#D85A30", linewidth=0.7)
    sm2 = _smooth(steps, window)
    ax3.plot(np.arange(len(steps) - len(sm2), len(steps)), sm2, color="#993C1D", linewidth=2)
    ax3.set_xlabel("Épisode")
    ax3.set_ylabel("Pas")
    ax3.set_title("Nombre de pas par épisode (↓ = meilleure politique)")
    ax3.grid(True, alpha=0.25)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Tableau de bord sauvegardé → {save_path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# Inspection de la Q-table
# ===========================================================================

def print_q_table_top(q_table: np.ndarray, n: int = 10) -> None:
    """
    Affiche les n états avec la Q-valeur maximale la plus élevée.

    Args:
        q_table : numpy array (n_states × n_actions)
        n       : nombre d'états à afficher
    """
    from src.environment import ACTION_LABELS

    max_vals = q_table.max(axis=1)
    top_states = np.argsort(max_vals)[-n:][::-1]

    print("\n  Top états par Q-valeur maximale :")
    print(f"  {'État':>6}  {'Meilleure action':<16}  {'Q-max':>8}")
    print("  " + "-" * 38)
    for s in top_states:
        best_a = int(np.argmax(q_table[s]))
        label = ACTION_LABELS.get(best_a, str(best_a))
        print(f"  {s:>6}  {label:<16}  {max_vals[s]:>8.2f}")
    print()
