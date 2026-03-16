import gymnasium as gym
from gymnasium.envs.registration import register

from .large_taxi import LargeTaxiEnv

register(
    id="LargeTaxi-v0",
    entry_point="src.envs:LargeTaxiEnv",
    max_episode_steps=500,
)

__all__ = ["LargeTaxiEnv"]
