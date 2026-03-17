import numpy as np
from env import SimpleTaxiEnv
from stable_baselines3 import DQN

print("Création de l'environnement...")
env = SimpleTaxiEnv()

print("Chargement du modèle...")
try:
    agent = DQN.load("models/dqn_taxi", env=env)
    print("✓ Modèle chargé avec succès")
except Exception as e:
    print(f"✗ Erreur lors du chargement: {e}")
    exit(1)

print("\nTest de prédiction (5 étapes)...")
obs, _ = env.reset()
for i in range(5):
    print(f"  Étape {i+1}...")
    try:
        action, _ = agent.predict(obs, deterministic=True)
        print(f"    Action prédite: {action}")
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"    Reward: {reward:.2f}, Done: {terminated or truncated}")
        if terminated or truncated:
            print("    Épisode terminé")
            break
    except Exception as e:
        print(f"    ✗ Erreur: {e}")
        exit(1)

print("\n✓ Test réussi")
env.close()
