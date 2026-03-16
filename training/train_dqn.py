"""
train_dqn.py
------------
Script d'entraînement DQN pour l'environnement Taxi 15×10.

Utilisation :
    python training/train_dqn.py --total_timesteps 300000 --learning_rate 3e-4
"""

import sys
import os
import argparse
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs import LargeTaxiEnv
from src.agent import DQNTaxiAgent


DEFAULTS = {
    "total_timesteps": 300_000,
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "batch_size": 128,
    "gamma": 0.99,
    "target_update_interval": 1000,
    "exploration_final_eps": 0.02,
    "exploration_fraction": 0.1,
    "log_dir": "logs",
    "checkpoint_dir": "training/checkpoints",
}


def train_dqn(cfg):
    """Lance l'entraînement DQN."""
    print("\n" + "=" * 60)
    print("  ENTRAÎNEMENT DQN — Taxi 15×10 Multi-Passagers")
    print("=" * 60)

    # Créer l'environnement
    env = LargeTaxiEnv(render_mode=None, grid_width=15, grid_height=10, n_passengers=2)
    print(f"\n  Environnement : {env.__class__.__name__}")
    print(f"  Grille : {env.grid_width}×{env.grid_height}")
    print(f"  Passagers : {env.n_passengers}")
    print(f"  Observation space : {env.observation_space}")
    print(f"  Action space : {env.action_space}\n")

    # Créer l'agent DQN
    agent = DQNTaxiAgent(
        env=env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        target_update_interval=cfg["target_update_interval"],
        exploration_final_eps=cfg["exploration_final_eps"],
        exploration_fraction=cfg["exploration_fraction"],
        tensorboard_log=cfg["log_dir"],
    )

    print(f"  Agent DQN créé")
    print(f"  Learning rate : {cfg['learning_rate']}")
    print(f"  Buffer size : {cfg['buffer_size']}")
    print(f"  Batch size : {cfg['batch_size']}")
    print(f"  Total timesteps : {cfg['total_timesteps']}\n")

    # Entraîner
    print(f"  Lancement de l'entraînement...")
    print(f"  TensorBoard : tensorboard --logdir {cfg['log_dir']}\n")

    agent.train(
        total_timesteps=cfg["total_timesteps"],
        checkpoint_dir=cfg["checkpoint_dir"],
    )

    # Sauvegarder le modèle final
    final_path = os.path.join(cfg["checkpoint_dir"], "dqn_taxi_final.zip")
    agent.save(final_path)

    env.close()
    print("\n" + "=" * 60)
    print("  Entraînement terminé !")
    print("=" * 60 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Entraînement DQN — Taxi 15×10")
    p.add_argument("--total_timesteps", type=int, default=DEFAULTS["total_timesteps"])
    p.add_argument("--learning_rate", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--buffer_size", type=int, default=DEFAULTS["buffer_size"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    p.add_argument("--target_update_interval", type=int, default=DEFAULTS["target_update_interval"])
    p.add_argument("--exploration_final_eps", type=float, default=DEFAULTS["exploration_final_eps"])
    p.add_argument("--exploration_fraction", type=float, default=DEFAULTS["exploration_fraction"])
    p.add_argument("--log_dir", type=str, default=DEFAULTS["log_dir"])
    p.add_argument("--checkpoint_dir", type=str, default=DEFAULTS["checkpoint_dir"])
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    train_dqn(cfg)
