"""
File: main
Use: Run CartPole subtask using Isaac Sim
Update: 
"""

import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ISAAC")
parser.add_argument("--mode", choices=["train", "eval"], required=True)
parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--epsilon", type=float, default=0.5)
parser.add_argument("--epsilon_min", type=float, default=0.01)
parser.add_argument("--epsilon_decay", type=float, default=0.97)
parser.add_argument("--iterations", type=int, default=50)
parser.add_argument("--steps_per_iter", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--obs_bins", type=int, nargs=4, default=[12, 12, 8, 12])
parser.add_argument("--save_dir", type=str, default="models")



AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app



import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401 (registers Isaac Lab envs)
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
from discrete import to_discrete
from agent import Agent



# Isaac Lab CartPole obs order: pole_pos, pole_vel, cart_pos, cart_vel
DEFAULT_OBS_RANGES = [(-0.21, 0.21), (-3.0, 3.0), (-2.4, 2.4), (-3.0, 3.0)]
# Discrete action mapping: 0 -> push left, 1 -> push right
ACTION_MAP = {0: -10.0, 1: 10.0}



def extract_all_obs(obs_dict) -> np.ndarray:
    """Extract numpy obs from Isaac Lab obs dict"""
    tensor = obs_dict["policy"]
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    return tensor


def collect_experience(env, agent: Agent, steps: int, num_envs: int, device: str):
    obs_dict, _ = env.reset()
    all_obs = extract_all_obs(obs_dict)
    states = agent.get_states(all_obs)  # vectorized discretization
    ep_rewards = np.zeros(num_envs)
    completed_rewards = []

    for _ in tqdm(range(steps), desc="collect", leave=False):

        actions_idx = agent.select_actions(states)
        forces = np.where(actions_idx == 0, -10.0, 10.0)
        actions_tensor = torch.tensor(forces, dtype=torch.float32, device=device).unsqueeze(1)

        obs_dict, rewards, terminated, truncated, _ = env.step(actions_tensor)

        rewards_np = rewards.cpu().numpy().flatten()

        term_np = terminated.cpu().numpy().flatten()
        trunc_np = truncated.cpu().numpy().flatten()

        all_obs = extract_all_obs(obs_dict)

        next_states = agent.get_states(all_obs)
        agent.update_model_batch(states, actions_idx, rewards_np, next_states)

        ep_rewards += rewards_np
        done_mask = term_np | trunc_np
        if done_mask.any():
            completed_rewards.extend(ep_rewards[done_mask].tolist())
            ep_rewards[done_mask] = 0

        states = next_states

    return completed_rewards if completed_rewards else [0.0]


def make_env(task: str, num_envs: int):
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = num_envs
    return gym.make(task, cfg=env_cfg)


def train(args):
    env = make_env(args.task, args.num_envs)
    device = "cuda"

    disc = to_discrete(args.obs_bins, DEFAULT_OBS_RANGES)
    agent = Agent(disc, n_actions=2, gamma=args.gamma, epsilon=args.epsilon)

    best_avg = -float("inf")

    for iteration in tqdm(range(args.iterations), desc="train"):
        rewards = collect_experience(env, agent, args.steps_per_iter, args.num_envs, device)
        avg = np.mean(rewards)

        # TODO: Figure out how to speed this up
        agent.solve()
        agent.epsilon = max(args.epsilon_min, agent.epsilon * args.epsilon_decay)

        print(f"Iter {iteration+1}/{args.iterations} | "
              f"R: {avg:.1f} | E: {agent.epsilon:.3f}")

        if avg > best_avg:
            best_avg = avg
            os.makedirs(args.save_dir, exist_ok=True)
            agent.save(
                os.path.join(args.save_dir, "best_policy.npy"),
                os.path.join(args.save_dir, "best_values.npy"),
            )
            print(f" {best_avg:.1f}")

    env.close()
    print(f"Best R: {best_avg:.1f}")


def evaluate(args):
    env = make_env(args.task, args.num_envs)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    disc = to_discrete(args.obs_bins, DEFAULT_OBS_RANGES)
    agent = Agent(disc, n_actions=2, gamma=args.gamma, epsilon=0.0)

    agent.load(
        os.path.join(args.save_dir, "best_policy.npy"),
        os.path.join(args.save_dir, "best_values.npy"),
    )

    rewards = collect_experience(env, agent, args.eval_steps, args.num_envs, device)

    env.close()
    print(f"R:{np.mean(rewards):.1f} | {len(rewards)} eps")


if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)

    simulation_app.close()
