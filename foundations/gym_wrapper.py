"""
Ref: https://gymnasium.farama.org/introduction/basic_usage/
"""

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

env = gym.make("CarRacing-v3", render_mode="human" )
print(env.action_space)
print(env.observation_space.sample)

wrapped_env = FlattenObservation(env)
print(wrapped_env.observation_space.shape)
