# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""
Modified From Isaac Drone Racer GitHub by Kousheek Chakraborty

File:   train.py
Use:    RL train script for maneuverability
Update: Fri, 20 Mar 2026

Usage: python3 scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs 4096

Changes:
- Changes to documentation
- Removal of ml_framework check
- Only use PyTorch
"""

import argparse
import os
import pickle
import random
import sys
import matplotlib
import matplotlib.pyplot as plot
from datetime import datetime

# Ensure this script's directory is resolved first so local utils/ isn't
# shadowed by IsaacLab's utils package on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add repo root so drfm package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import skrl
from packaging import version


def plot_training_curves(log_dir: str) -> None:
    """Generate reward and episode-length curves from skrl's CSV progress log.

    Args:
        log_dir: Path to the skrl experiment directory (contains ``*.csv``).
    """
    import glob

    matplotlib.use("Agg")

    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not csv_files:
        print(f"[INFO] No CSV logs found in {log_dir} — skipping training curve plots.")
        return

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        reward_cols = [c for c in df.columns if "reward" in c.lower()]
        length_cols = [c for c in df.columns if "length" in c.lower()]
        timestep_col = next((c for c in df.columns if "timestep" in c.lower()), None)

        plot_cols = reward_cols + length_cols
        if not plot_cols:
            continue

        x = df[timestep_col] if timestep_col else np.arange(len(df))
        fig, axes = plt.subplots(len(plot_cols), 1, figsize=(10, 3 * len(plot_cols)), squeeze=False)

        for i, col in enumerate(plot_cols):
            axes[i, 0].plot(x, df[col], linewidth=1.2)
            axes[i, 0].set_ylabel(col)
            axes[i, 0].set_xlabel("Timestep")
            axes[i, 0].grid(True, alpha=0.4)

        plt.suptitle(os.path.basename(log_dir), fontsize=9)
        plt.tight_layout()
        out = csv_path.replace(".csv", "_curves.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[INFO] Training curves saved: {out}")


# Parse command-line arguments (before importing Isaac modules)
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Task name (e.g., Isaac-Drone-Racer-v0).")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
parser.add_argument("--distributed", action="store_true", default=False, help="Enable multi-GPU training.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for resuming training.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training iterations.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="RL algorithm to use.",
)
parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
parser.add_argument("--wandb_project", type=str, default="drone-racer", help="W&B project name.")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# If recording
if args_cli.video:
    args_cli.enable_cameras = True

# Pass remaining arguments to Hydra for configuration
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import drfm.envs.isaac  # noqa: F401

from skrl.utils.runner.torch import Runner

# Select learning algorithm, defaults to PPO
algorithm = args_cli.algorithm.lower()
if algorithm == "ppo":
    agent_cfg_entry_point = "skrl_cfg_entry_point"
else:
    agent_cfg_entry_point = f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
) -> None:
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Configure training iterations
    if args_cli.max_iterations is not None:
        total_timesteps = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        agent_cfg["trainer"]["timesteps"] = total_timesteps

    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]

    # Logging
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment to: {log_root_path}")
    print(f"[INFO] TensorBoard: tensorboard --logdir {log_root_path}")

    # Build experiment name: timestamp + algorithm + custom name
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir += f"_{algorithm}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"

    # Update configuration with actual logging paths
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    if args_cli.wandb:
        agent_cfg["agent"]["experiment"]["wandb"] = True
        agent_cfg["agent"]["experiment"]["wandb_kwargs"] = {
            "project": args_cli.wandb_project,
            "name": os.path.basename(log_dir),
        }

    # Save configurations and parameters to disk
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)

    # Load checkpoint if resuming training
    resume_path = None
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    # Create the training environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Convert multi-agent environments to single-agent for PPO
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)

    # Setup video recording wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(env, agent_cfg)

    # If loading from trained
    if resume_path:
        print(f"[INFO] Resuming training from checkpoint: {resume_path}")
        runner.agent.load(resume_path)

    runner.run()
    env.close()

    plot_training_curves(log_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
