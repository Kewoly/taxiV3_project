from stable_baselines3 import DQN
from env import SimpleTaxiEnv
import numpy as np


def evaluate_agent(model_path="models/dqn_taxi", n_episodes=200):
    """Évalue un agent DQN entraîné."""

    env = SimpleTaxiEnv()
    agent = DQN.load(model_path, env=env)

    rewards = []
    steps = []
    successes = 0

    print("\n" + "="*60)
    print(f"ÉVALUATION - {n_episodes} épisodes")
    print("="*60 + "\n")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        
        rewards.append(total_reward)
        steps.append(step_count)
        if terminated:
            successes += 1
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{n_episodes} - Succès: {successes}/{ep + 1}")
    
    success_rate = successes / n_episodes * 100
    
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    print(f"Taux de succès        : {success_rate:.1f}%")
    print(f"Récompense moyenne    : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Pas moyens            : {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"Min/Max récompense    : {np.min(rewards):.0f} / {np.max(rewards):.0f}")
    print(f"Min/Max pas           : {np.min(steps)} / {np.max(steps)}")
    print("="*60 + "\n")
    
    env.close()
    
    return success_rate, np.mean(rewards), np.mean(steps)


if __name__ == "__main__":
    evaluate_agent()
