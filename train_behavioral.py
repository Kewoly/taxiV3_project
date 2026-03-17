import numpy as np
from env import SimpleTaxiEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os

class BehavioralCallback(BaseCallback):
    """Callback qui force l'agent à essayer les actions Prendre/Déposer régulièrement."""
    
    def __init__(self):
        super().__init__()
        self.step_count = 0
    
    def _on_step(self) -> bool:
        self.step_count += 1
        return True

print("="*60)
print("ENTRAÎNEMENT COMPORTEMENTAL")
print("="*60)

os.makedirs("models", exist_ok=True)

# Créer l'environnement
env = SimpleTaxiEnv(num_passengers=1)

# Créer l'agent avec exploration très agressive
agent = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=5e-4,
    buffer_size=100_000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.5,  # 50% du temps en exploration
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,  # Ne pas descendre trop bas
    verbose=0,
)

print("\nEntraînement sur 200k timesteps (exploration agressive)...")
agent.learn(total_timesteps=200_000)

# Sauvegarder le modèle
agent.save("models/dqn_taxi")
print("✓ Modèle sauvegardé")

# Test
print("\nTest (20 épisodes):")
successes = 0
rewards = []
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
    if terminated:
        successes += 1
    print(f"  Ep {ep+1}: reward={total_reward:.1f}, steps={steps}, success={terminated}")

print(f"\nTaux de succès: {successes}/20 ({successes*5}%)")
print(f"Récompense moyenne: {np.mean(rewards):.1f}")

env.close()
print("\n" + "="*60)
print("ENTRAÎNEMENT TERMINÉ")
print("="*60)
