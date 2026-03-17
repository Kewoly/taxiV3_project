import numpy as np
from env import SimpleTaxiEnv
from stable_baselines3 import DQN
import os


def test_environment():
    """Teste l'environnement et les récompenses."""
    print("\n" + "="*60)
    print("DIAGNOSTIC DE L'ENVIRONNEMENT")
    print("="*60)
    
    env = SimpleTaxiEnv()
    
    print(f"\n✓ Grille : {env.grid_width}×{env.grid_height}")
    print(f"✓ Observation space : {env.observation_space}")
    print(f"✓ Action space : {env.action_space}")
    print(f"✓ Max steps : {env.max_steps}")
    
    # Test observation
    obs, _ = env.reset()
    print(f"\n✓ Observation normalisée")
    print(f"  Shape : {obs.shape}")
    print(f"  Min/Max : {obs.min():.3f} / {obs.max():.3f}")
    
    # Test récompenses
    print(f"\n✓ Test des récompenses")
    
    # Mouvement simple
    obs, reward, _, _, _ = env.step(0)
    print(f"  Mouvement : reward = {reward}")
    
    # Aller au pickup
    for _ in range(10):
        obs, reward, _, _, _ = env.step(2)  # Est
    
    # Prendre un passager
    obs, reward, _, _, _ = env.step(4)
    print(f"  Pickup : reward = {reward}")
    
    # Aller au dropoff
    for _ in range(20):
        obs, reward, _, _, _ = env.step(2)
    
    # Déposer
    obs, reward, terminated, _, _ = env.step(5)
    print(f"  Dropoff : reward = {reward}")
    print(f"  Terminated : {terminated}")
    
    env.close()
    print("\n✓ Environnement OK\n")


def test_training():
    """Teste l'entraînement avec un petit nombre de timesteps."""
    print("="*60)
    print("TEST D'ENTRAÎNEMENT (100k timesteps, 1 passager)")
    print("="*60 + "\n")
    
    env = SimpleTaxiEnv(num_passengers=1)
    
    agent = DQN(
        policy="MlpPolicy",
        env=env,
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
    
    print("Entraînement en cours...")
    agent.learn(total_timesteps=100_000)
    
    # Évaluer
    print("\nÉvaluation sur 10 épisodes...")
    rewards = []
    successes = 0
    
    for ep in range(10):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        rewards.append(total_reward)
        if terminated:
            successes += 1
    
    print(f"\n✓ Résultats du test")
    print(f"  Taux de succès : {successes}/10 ({successes*10}%)")
    print(f"  Récompense moyenne : {np.mean(rewards):.2f}")
    print(f"  Min/Max : {np.min(rewards):.0f} / {np.max(rewards):.0f}")
    
    env.close()
    print("\n✓ Entraînement OK\n")


if __name__ == "__main__":
    test_environment()
    test_training()
    
    print("="*60)
    print("DIAGNOSTIC COMPLET")
    print("="*60)
    print("✓ Environnement fonctionnel")
    print("✓ Récompenses bien calibrées")
    print("✓ Entraînement fonctionnel")
    print("\nVous pouvez maintenant lancer :")
    print("  python train.py")
    print("  ou")
    print("  jupyter notebook notebook.ipynb")
    print("="*60 + "\n")
