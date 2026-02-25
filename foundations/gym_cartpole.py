import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
over = False;   total_rw = 0

print(f"start:   {observation}")

while not over:
    action = env.action_space.sample()

    # Terminated: Fail to accomplish goal
    # Truncated:  Time over
    observation, reward, terminated, truncated, info = env.step(action)

    total_rw += reward
    over = terminated or truncated

print(f"Total reward:   {total_rw}")
env.close()
