"""
agent.py
--------
Agent DQN pour l'environnement Taxi custom 15×10.
Utilise Stable Baselines 3 pour l'entraînement deep RL.
"""

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os


class DQNTaxiAgent:
    """Agent DQN pour Taxi 15×10 multi-passagers."""

    def __init__(self, env, learning_rate=3e-4, buffer_size=100_000, batch_size=128,
                 gamma=0.99, target_update_interval=1000, exploration_final_eps=0.02,
                 exploration_fraction=0.1, tensorboard_log="./logs/"):
        """
        Initialise l'agent DQN.

        Args:
            env : environnement Gymnasium
            learning_rate : taux d'apprentissage du réseau
            buffer_size : taille du replay buffer
            batch_size : taille des batches
            gamma : facteur d'actualisation
            target_update_interval : fréquence de mise à jour du réseau cible
            exploration_final_eps : epsilon final (exploration)
            exploration_fraction : fraction des timesteps pour l'exploration
            tensorboard_log : dossier pour les logs TensorBoard
        """
        self.env = env
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    def train(self, total_timesteps=300_000, checkpoint_dir="./training/checkpoints/"):
        """Lance l'entraînement DQN avec sauvegarde automatique."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=checkpoint_dir,
            name_prefix="dqn_taxi",
            save_replay_buffer=False,
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )

    def save(self, path):
        """Sauvegarde le modèle DQN."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"[DQN] Modèle sauvegardé → {path}")

    def load(self, path):
        """Charge un modèle DQN sauvegardé."""
        self.model = DQN.load(path, env=self.env)
        print(f"[DQN] Modèle chargé ← {path}")

    def predict(self, obs, deterministic=True):
        """Prédit une action."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)
