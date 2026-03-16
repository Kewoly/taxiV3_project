import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from env import SimpleTaxiEnv

# Créer l'environnement
env = SimpleTaxiEnv()

print("="*60)
print("ENTRAÎNEMENT DQN - VERSION CORRIGÉE")
print("="*60)
print("\nChangements appliqués:")
print("  ✓ Récompense step: -1.0 → -0.1 (moins pénalisant)")
print("  ✓ Récompense pickup: 5.0 → 10.0 (plus encourageant)")
print("  ✓ Récompense dropoff: 20.0 → 50.0 (bien plus encourageant)")
print("  ✓ Learning rate: 1e-3 → 1e-3 (maintenu)")
print("  ✓ Batch size: 128 → 64 (plus stable)")
print("="*60 + "\n")

agent = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-1,
    buffer_size=100_000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.15,
    exploration_final_eps=0.05,
    target_update_interval=1000,
    verbose=1,
    tensorboard_log="./logs/",
)

print("Entraînement en cours... (300,000 timesteps)\n")
agent.learn(total_timesteps=300_000)

os.makedirs("models", exist_ok=True)
agent.save("models/dqn_taxi_fixed")
print("\n✓ Modèle sauvegardé → models/dqn_taxi_fixed.zip")

# Évaluation rapide
print("\n" + "="*60)
print("ÉVALUATION RAPIDE")
print("="*60)

total_reward = 0
success_count = 0
for episode in range(10):
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    
    total_reward += episode_reward
    if env.delivered == 2:
        success_count += 1
    
    print(f"Episode {episode+1}: Reward={episode_reward:.1f}, Delivered={env.delivered}/2")

print(f"\nMoyenne: {total_reward/10:.1f}")
print(f"Taux de succès: {success_count}/10 ({success_count*10}%)")

env.close()
