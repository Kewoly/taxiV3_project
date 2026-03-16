"""
train.py
--------
Script principal d'entraînement de l'agent Q-Learning sur Taxi-v3.

Utilisation :
    python training/train.py
    python training/train.py --episodes 2000 --alpha 0.15 --gamma 0.99

Sorties :
  - training/checkpoints/qtable_ep<N>.pkl  : sauvegardes intermédiaires
  - training/checkpoints/qtable_final.pkl  : Q-table finale
  - logs/                                  : métriques pour TensorBoard
  - logs/rewards.png                       : courbe de récompenses
"""

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

# Ajout du dossier racine au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import make_env, describe_env
from src.agent import QLearningAgent
from src.interface import plot_all

# TensorBoard (optionnel mais recommandé)
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    try:
        from tensorboard.summary.writer.event_file_writer import EventFileWriter  # noqa
        from torch.utils.tensorboard import SummaryWriter
        TB_AVAILABLE = True
    except ImportError:
        TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paramètres par défaut
# ---------------------------------------------------------------------------

DEFAULTS = {
    "episodes":      2000,
    "max_steps":     200,
    "alpha":         0.1,
    "gamma":         0.99,
    "epsilon":       1.0,
    "epsilon_min":   0.01,
    "epsilon_decay": 0.995,
    "checkpoint_every": 500,
    "log_dir":       "logs",
    "checkpoint_dir": "training/checkpoints",
}


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def train(cfg: dict) -> QLearningAgent:
    """
    Lance la boucle d'entraînement Q-Learning.

    Args:
        cfg : dictionnaire de configuration (voir DEFAULTS)

    Returns:
        agent entraîné
    """
    describe_env()

    # Environnement
    env = make_env(render_mode=None)
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    # Agent
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
    )
    print(f"\n  {agent}\n")

    # TensorBoard
    writer = None
    if TB_AVAILABLE:
        os.makedirs(cfg["log_dir"], exist_ok=True)
        writer = SummaryWriter(log_dir=cfg["log_dir"])
        print(f"  TensorBoard : tensorboard --logdir {cfg['log_dir']}\n")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # Historiques
    all_rewards  = []
    all_epsilons = []
    all_steps    = []

    # --------------- boucle principale ---------------
    pbar = tqdm(range(1, cfg["episodes"] + 1), desc="Entraînement", unit="ep")

    for episode in pbar:
        state, _ = env.reset()
        total_reward = 0

        for step in range(cfg["max_steps"]):
            action     = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.update(state, action, reward, next_state, terminated or truncated)

            state        = next_state
            total_reward += reward

            if terminated or truncated:
                break

        # Décroissance epsilon
        agent.decay_epsilon()

        # Historique
        all_rewards.append(total_reward)
        all_epsilons.append(agent.epsilon)
        all_steps.append(step + 1)

        # Logging TensorBoard
        if writer:
            writer.add_scalar("Reward/episode",  total_reward,   episode)
            writer.add_scalar("Epsilon/episode", agent.epsilon,  episode)
            writer.add_scalar("Steps/episode",   step + 1,       episode)

        # Barre de progression
        if episode % 50 == 0:
            mean_r = np.mean(all_rewards[-50:])
            pbar.set_postfix({"reward_50": f"{mean_r:.1f}", "ε": f"{agent.epsilon:.3f}"})

        # Sauvegarde intermédiaire
        if episode % cfg["checkpoint_every"] == 0:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"qtable_ep{episode}.pkl")
            agent.save(ckpt_path)

    # Sauvegarde finale
    final_path = os.path.join(cfg["checkpoint_dir"], "qtable_final.pkl")
    agent.save(final_path)

    env.close()
    if writer:
        writer.close()

    # Résumé
    print(f"\n  Entraînement terminé sur {cfg['episodes']} épisodes.")
    print(f"  Reward moyen (200 derniers) : {np.mean(all_rewards[-200:]):.2f}")
    print(f"  Epsilon final               : {agent.epsilon:.4f}")

    # Graphiques
    plot_path = os.path.join(cfg["log_dir"], "training_dashboard.png")
    plot_all(all_rewards, all_epsilons, all_steps, window=50, save_path=plot_path)

    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Entraînement Q-Learning — Taxi-v3")
    p.add_argument("--episodes",       type=int,   default=DEFAULTS["episodes"])
    p.add_argument("--alpha",          type=float, default=DEFAULTS["alpha"])
    p.add_argument("--gamma",          type=float, default=DEFAULTS["gamma"])
    p.add_argument("--epsilon",        type=float, default=DEFAULTS["epsilon"])
    p.add_argument("--epsilon_min",    type=float, default=DEFAULTS["epsilon_min"])
    p.add_argument("--epsilon_decay",  type=float, default=DEFAULTS["epsilon_decay"])
    p.add_argument("--checkpoint_every", type=int, default=DEFAULTS["checkpoint_every"])
    p.add_argument("--log_dir",        type=str,   default=DEFAULTS["log_dir"])
    p.add_argument("--checkpoint_dir", type=str,   default=DEFAULTS["checkpoint_dir"])
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
