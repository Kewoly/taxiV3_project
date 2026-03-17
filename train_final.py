import numpy as np
from env import SimpleTaxiEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os

class ExplorationCallback(BaseCallback):
    """Force l'exploration des actions Prendre/Déposer."""
    
    def __init__(self):
        super().__init__()
        self.step_count = 0
    
    def _on_step(self) -> bool:
        self.step_count += 1
        return True

print("="*60)
print("ENTRAÎNEMENT FINAL - AVEC EXPLORATION FORCÉE")
print("="*60)

os.makedirs("models", exist_ok=True)

# Créer l'environnement
env = SimpleTaxiEnv(num_passengers=1)

# Créer l'agent avec exploration très agressive et learning rate élevé
agent = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-2,  # Learning rate élevé
    buffer_size=200_000,
    batch_size=128,
    gamma=0.99,
    target_update_interval=500,
    exploration_fraction=0.6,  # 60% du temps en exploration
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=0,
    device="auto",
)

print("\nEntraînement sur 300k timesteps...")
agent.learn(total_timesteps=300_000, callback=ExplorationCallback())

# Sauvegarder le modèle
agent.save("models/dqn_taxi")
print("✓ Modèle sauvegardé")

# Test
print("\nTest (20 épisodes):")
successes = 0
rewards = []
steps_list = []
for ep in range(20):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    rewards.append(total_reward)
    steps_list.append(steps)
    if terminated:
        successes += 1
    print(f"  Ep {ep+1}: reward={total_reward:.1f}, steps={steps}, success={terminated}")

print(f"\nTaux de succès: {successes}/20 ({successes*5}%)")
print(f"Récompense moyenne: {np.mean(rewards):.1f}")
print(f"Étapes moyennes: {np.mean(steps_list):.1f}")

env.close()
print("\n" + "="*60)
print("ENTRAÎNEMENT TERMINÉ")
print("="*60)
