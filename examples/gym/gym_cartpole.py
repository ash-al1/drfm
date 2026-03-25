"""
Ref: https://gymnasium.farama.org/introduction/basic_usage/
"""

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

print("-"*80)
print(f"action-space: {env.action_space}")
print(f"sample-action: {env.action_space.sample()}")
print(f"observation-space: {env.observation_space}")
print(f"observation-sample: {env.observation_space.sample()}")
print("-"*80)

observation, info = env.reset()
over = False;   total_rw = 0

print(f"start:   {observation}")

while not over:
    action = env.action_space.sample()

    # Terminated: Fail/succeed in accomplishing goal
    # Truncated:  Time over
    observation, reward, terminated, truncated, info = env.step(action)

    total_rw += reward
    over = terminated or truncated

print(f"Total reward:   {total_rw}")
env.close()
