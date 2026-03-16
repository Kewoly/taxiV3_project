"""
evaluate.py
-----------
Script de test final : charge un agent entraîné, joue N épisodes
et génère les métriques de performance.

Utilisation :
    python evaluation/evaluate.py
    python evaluation/evaluate.py --model training/checkpoints/qtable_final.pkl --episodes 100
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import make_env
from src.agent import QLearningAgent
from src.interface import render_episode, print_q_table_top


# ---------------------------------------------------------------------------
# Évaluation silencieuse (sans rendu)
# ---------------------------------------------------------------------------

def evaluate_silent(agent: QLearningAgent, n_episodes: int = 100, max_steps: int = 200) -> dict:
    """
    Joue n_episodes épisodes sans rendu et collecte les métriques.

    Args:
        agent      : agent chargé (epsilon mis à 0 pour évaluation pure)
        n_episodes : nombre d'épisodes de test
        max_steps  : pas maximum par épisode

    Returns:
        dict de métriques : mean_reward, std_reward, success_rate,
                            mean_steps, min_reward, max_reward
    """
    env = make_env(render_mode=None)

    # Désactiver l'exploration pour l'évaluation
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    steps_list = []
    successes = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if terminated or truncated:
                if reward == 20:
                    successes += 1
                break

        rewards.append(total_reward)
        steps_list.append(step + 1)

    env.close()
    agent.epsilon = saved_epsilon

    metrics = {
        "mean_reward":   float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "min_reward":    float(np.min(rewards)),
        "max_reward":    float(np.max(rewards)),
        "success_rate":  successes / n_episodes * 100,
        "mean_steps":    float(np.mean(steps_list)),
        "n_episodes":    n_episodes,
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Affiche les métriques dans la console."""
    print("\n" + "=" * 50)
    print("  RÉSULTATS D'ÉVALUATION")
    print("=" * 50)
    print(f"  Épisodes testés     : {metrics['n_episodes']}")
    print(f"  Reward moyen        : {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Reward min / max    : {metrics['min_reward']:.0f} / {metrics['max_reward']:.0f}")
    print(f"  Taux de succès      : {metrics['success_rate']:.1f} %")
    print(f"  Pas moyens          : {metrics['mean_steps']:.1f}")
    print("=" * 50 + "\n")


def plot_eval_distribution(rewards: list, save_path: str = None) -> None:
    """Trace la distribution des récompenses d'évaluation."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rewards, bins=20, color="#534AB7", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(rewards), color="#D85A30", linewidth=2, label=f"Moyenne : {np.mean(rewards):.1f}")
    ax.set_xlabel("Récompense totale")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des récompenses — évaluation finale")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Eval] Distribution sauvegardée → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Évaluation de l'agent Taxi-v3")
    p.add_argument(
        "--model",
        type=str,
        default="training/checkpoints/qtable_final.pkl",
        help="Chemin vers la Q-table sauvegardée",
    )
    p.add_argument("--episodes",  type=int,  default=100,   help="Nombre d'épisodes d'évaluation")
    p.add_argument("--demo",      action="store_true",       help="Joue 1 épisode avec rendu visuel")
    p.add_argument("--demo_steps",type=int,  default=200,   help="Pas max pour la démo")
    p.add_argument("--delay",     type=float, default=0.2,  help="Délai entre pas (démo)")
    p.add_argument("--top_states",action="store_true",       help="Affiche les top états Q-table")
    p.add_argument("--save_plot", type=str,  default="logs/eval_distribution.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Chargement de l'agent
    env_tmp = make_env()
    agent = QLearningAgent(
        n_states=env_tmp.observation_space.n,
        n_actions=env_tmp.action_space.n,
    )
    env_tmp.close()

    if not os.path.exists(args.model):
        print(f"[Erreur] Modèle introuvable : {args.model}")
        print("  Lancez d'abord l'entraînement : python training/train.py")
        sys.exit(1)

    agent.load(args.model)
    agent.epsilon = 0.0  # mode greedy pur

    # Inspection Q-table
    if args.top_states:
        print_q_table_top(agent.q_table, n=15)

    # Évaluation silencieuse
    print(f"\n[Eval] Évaluation sur {args.episodes} épisodes...")
    metrics = evaluate_silent(agent, n_episodes=args.episodes)
    print_metrics(metrics)

    # Distribution
    env_tmp2 = make_env()
    agent.epsilon = 0.0
    rewards_list = []
    for _ in range(args.episodes):
        s, _ = env_tmp2.reset()
        tot = 0
        for _ in range(200):
            a = agent.select_action(s)
            s, r, te, tr, _ = env_tmp2.step(a)
            tot += r
            if te or tr:
                break
        rewards_list.append(tot)
    env_tmp2.close()
    plot_eval_distribution(rewards_list, save_path=args.save_plot)

    # Démo visuelle
    if args.demo:
        print("\n[Démo] Lancement d'un épisode avec rendu...\n")
        demo_env = make_env(render_mode="human")
        render_episode(demo_env, agent, max_steps=args.demo_steps, delay=args.delay)
        demo_env.close()
