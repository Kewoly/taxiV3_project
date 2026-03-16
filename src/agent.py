"""
agent.py
--------
Définition des agents RL pour Taxi-v3.

Deux implémentations disponibles :
  - QLearningAgent : Q-table classique (recommandé pour Taxi-v3, espace discret)
  - DQNAgent       : Deep Q-Network via Stable Baselines 3 (pour comparaison)

L'espace d'état discret de Taxi-v3 (500 états) rend le Q-Learning
particulièrement adapté : convergence rapide, interprétable, léger.
"""

import numpy as np
import os
import pickle

# DQN via Stable Baselines 3 (import optionnel)
try:
    from stable_baselines3 import DQN as SB3_DQN
    from stable_baselines3.common.callbacks import CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


# ===========================================================================
# Q-Learning Agent (table)
# ===========================================================================

class QLearningAgent:
    """
    Agent Q-Learning avec table Q de taille (n_states × n_actions).

    Hyperparamètres :
        alpha   : taux d'apprentissage        (défaut 0.1)
        gamma   : facteur d'actualisation     (défaut 0.99)
        epsilon : taux d'exploration initial  (défaut 1.0)
        epsilon_min  : plancher epsilon       (défaut 0.01)
        epsilon_decay: décroissance par épisode (défaut 0.995)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Table Q initialisée à zéro
        self.q_table = np.zeros((n_states, n_actions))

    # ------------------------------------------------------------------
    # Sélection d'action (epsilon-greedy)
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """
        Choisit une action selon la politique epsilon-greedy.

        Args:
            state : état courant (entier)

        Returns:
            action : entier dans [0, n_actions[
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)   # exploration
        return int(np.argmax(self.q_table[state]))      # exploitation

    # ------------------------------------------------------------------
    # Mise à jour Q-Learning (Bellman)
    # ------------------------------------------------------------------

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Applique la règle de mise à jour Q-Learning :
            Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]

        Args:
            state      : état courant
            action     : action prise
            reward     : récompense reçue
            next_state : état suivant
            done       : True si l'épisode est terminé
        """
        best_next = 0.0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    # ------------------------------------------------------------------
    # Décroissance epsilon
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Réduit epsilon après chaque épisode (exploration → exploitation)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Sauvegarde la Q-table et les hyperparamètres dans un fichier .npy."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "q_table":       self.q_table,
            "epsilon":       self.epsilon,
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon_min":   self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Agent] Q-table sauvegardée → {path}")

    def load(self, path: str) -> None:
        """Charge une Q-table depuis un fichier .npy."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table       = data["q_table"]
        self.epsilon       = data["epsilon"]
        self.alpha         = data["alpha"]
        self.gamma         = data["gamma"]
        self.epsilon_min   = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]
        print(f"[Agent] Q-table chargée ← {path}")

    def __repr__(self) -> str:
        return (
            f"QLearningAgent("
            f"α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon:.3f}, "
            f"table={self.q_table.shape})"
        )


# ===========================================================================
# DQN Agent (Stable Baselines 3) — espace continu ou comparaison
# ===========================================================================

class DQNAgent:
    """
    Agent DQN utilisant Stable Baselines 3.
    Adapté si l'on veut comparer avec une approche Deep RL.

    Note : Pour Taxi-v3 (500 états discrets), Q-Learning est plus efficace.
           DQN est inclus à titre de comparaison pédagogique.
    """

    def __init__(self, env, tensorboard_log: str = "./logs/"):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 n'est pas installé. "
                "Lancez : pip install stable-baselines3"
            )
        self.env = env
        self.model = SB3_DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            target_update_interval=500,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    def train(self, total_timesteps: int = 100_000, checkpoint_dir: str = "./training/checkpoints/") -> None:
        """Lance l'entraînement DQN avec sauvegarde automatique."""
        callback = CheckpointCallback(
            save_freq=10_000,
            save_path=checkpoint_dir,
            name_prefix="dqn_taxi",
        )
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path: str) -> None:
        self.model.save(path)
        print(f"[DQN] Modèle sauvegardé → {path}")

    def load(self, path: str) -> None:
        self.model = SB3_DQN.load(path, env=self.env)
        print(f"[DQN] Modèle chargé ← {path}")

    def select_action(self, state) -> int:
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)
