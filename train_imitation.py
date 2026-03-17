import numpy as np
from env import SimpleTaxiEnv
from stable_baselines3 import DQN
import os

def simple_policy(env):
    """Politique simple basée sur des règles: naviguer vers passager, prendre, naviguer vers destination, déposer."""
    obs = env._get_obs()
    
    # Extraire les informations de l'observation
    taxi_x, taxi_y = env.taxi_x, env.taxi_y
    
    # Trouver le passager actif non embarqué
    for p in env.passengers:
        if p.get("active") and not p["in_taxi"]:
            # Aller vers le passager
            if taxi_x < p["x"]:
                return 2  # Est
            elif taxi_x > p["x"]:
                return 3  # Ouest
            elif taxi_y < p["y"]:
                return 0  # Sud
            elif taxi_y > p["y"]:
                return 1  # Nord
            else:
                # On est sur le passager, le prendre
                return 4  # Prendre
    
    # Si un passager est dans le taxi, aller vers sa destination
    for p in env.passengers:
        if p["in_taxi"]:
            if taxi_x < p["dest_x"]:
                return 2  # Est
            elif taxi_x > p["dest_x"]:
                return 3  # Ouest
            elif taxi_y < p["dest_y"]:
                return 0  # Sud
            elif taxi_y > p["dest_y"]:
                return 1  # Nord
            else:
                # On est à la destination, déposer
                return 5  # Déposer
    
    # Par défaut, action aléatoire
    return env.action_space.sample()

print("="*60)
print("ENTRAÎNEMENT PAR IMITATION (BEHAVIORAL CLONING)")
print("="*60)

os.makedirs("models", exist_ok=True)

# Phase 1: Générer des données avec la politique simple
print("\n[Phase 1] Collecte de données avec politique simple")
print("-" * 60)

env = SimpleTaxiEnv(num_passengers=1)

# Créer l'agent DQN
agent = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=100_000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.0,  # Pas d'exploration aléatoire
    exploration_initial_eps=0.0,
    exploration_final_eps=0.0,
    verbose=0,
)

# Remplir le replay buffer avec la politique simple
print("Remplissage du replay buffer avec politique simple (50k steps)...")
obs, _ = env.reset()
obs_prev = obs.copy()
for step in range(50_000):
    action = simple_policy(env)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    # Ajouter à la replay buffer (API correcte)
    agent.replay_buffer.add(
        obs_prev,
        obs,
        np.array([action]),  # Convertir en array
        reward,
        terminated or truncated,
        infos={}
    )
    
    obs_prev = obs.copy()
    
    if terminated or truncated:
        obs, _ = env.reset()
        obs_prev = obs.copy()
    
    if (step + 1) % 10_000 == 0:
        print(f"  {step + 1}/50_000 steps")

print("✓ Replay buffer rempli")

# Phase 2: Entraîner l'agent sur les données collectées
print("\n[Phase 2] Entraînement sur les données collectées (100k timesteps)")
print("-" * 60)

agent.learn(total_timesteps=100_000)

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
