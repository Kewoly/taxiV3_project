import numpy as np
from env import SimpleTaxiEnv
from stable_baselines3 import DQN
import os

class CuriosityEnv(SimpleTaxiEnv):
    """Wrapper qui ajoute une curiosité intrinsèque pour découvrir les actions Prendre/Déposer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_tried = [False] * 6  # Suivi des actions essayées
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.action_tried = [False] * 6
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Bonus de curiosité pour essayer les actions Prendre/Déposer
        if action in [4, 5] and not self.action_tried[action]:
            reward += 5.0  # Bonus pour première tentative
            self.action_tried[action] = True
        
        return obs, reward, terminated, truncated, info

print("="*60)
print("ENTRAÎNEMENT AVEC CURIOSITÉ INTRINSÈQUE")
print("="*60)

os.makedirs("models", exist_ok=True)

# Phase 1: Taxi sur passager avec bonus de curiosité
print("\n[Phase 1] Taxi sur passager + curiosité (50k timesteps)")
print("-" * 60)

env1 = CuriosityEnv(num_passengers=1)
original_reset = env1.reset

def reset_on_passenger(seed=None, options=None):
    obs, info = original_reset(seed=seed, options=options)
    if len(env1.passengers) > 0:
        env1.taxi_x = env1.passengers[0]["x"]
        env1.taxi_y = env1.passengers[0]["y"]
    return obs, info

env1.reset = reset_on_passenger

agent = DQN(
    policy="MlpPolicy",
    env=env1,
    learning_rate=1e-3,
    buffer_size=50_000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=0,
)

agent.learn(total_timesteps=50_000)

# Test phase 1
print("\nTest phase 1 (5 épisodes):")
successes = 0
for ep in range(5):
    obs, _ = env1.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env1.step(action)
        total_reward += reward
        done = terminated or truncated
    if terminated:
        successes += 1
    print(f"  Ep {ep+1}: reward={total_reward:.1f}, success={terminated}")

print(f"Taux de succès phase 1: {successes}/5")
env1.close()

# Phase 2: Taxi à distance normale (sans curiosité)
print("\n[Phase 2] Distance normale (50k timesteps)")
print("-" * 60)

env2 = SimpleTaxiEnv(num_passengers=1)
agent.set_env(env2)
agent.learn(total_timesteps=50_000, reset_num_timesteps=False)

# Test phase 2
print("\nTest phase 2 (10 épisodes):")
successes = 0
rewards = []
for ep in range(10):
    obs, _ = env2.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env2.step(action)
        total_reward += reward
        done = terminated or truncated
    rewards.append(total_reward)
    if terminated:
        successes += 1
    print(f"  Ep {ep+1}: reward={total_reward:.1f}, success={terminated}")

print(f"\nTaux de succès phase 2: {successes}/10 ({successes*10}%)")
print(f"Récompense moyenne: {np.mean(rewards):.1f}")

# Sauvegarder le modèle final
agent.save("models/dqn_taxi")
print("\n✓ Modèle sauvegardé → models/dqn_taxi")

env2.close()
print("\n" + "="*60)
print("ENTRAÎNEMENT TERMINÉ")
print("="*60)
