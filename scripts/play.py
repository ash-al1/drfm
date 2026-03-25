# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""
Modified From Isaac Drone Racer GitHub by Kousheek Chakraborty

File:   play.py
Use:    Visualize trained maneuverability
Update: Fri, 20 Mar 2026

Usage:
  python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1
  python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1 --video
  python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1 --checkpoint logs/skrl/.../model.pt

Changes:
+ Changes to documentation
+ Removal of ml_framework check
+ Only use PyTorch
"""

import argparse
import os
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

from isaaclab.app import AppLauncher

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Inference for trained drone racing agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Task name (e.g., Isaac-Drone-Racer-Play-v0).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use pretrained checkpoint from Nucleus (if available).",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="RL algorithm used for training.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Sync playback to real-time.")
parser.add_argument(
    "--renderer",
    type=str,
    default="RayTracedLighting",
    choices=["RayTracedLighting", "PathTracing"],
    help="Renderer to use.",
)
parser.add_argument("--log", type=int, default=None, help="Number of episodes to log (metrics only).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# If enable cameras
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)

import environment  # noqa: F401
from utils.logger import CSVLogger

# Import PyTorch runner
from skrl.utils.runner.torch import Runner

# Determine agent config entry point based on algorithm
algorithm = args_cli.algorithm.lower()


def main() -> None:
    # Validate logging configuration
    if args_cli.log and args_cli.num_envs > 1:
        raise ValueError("Logging is only supported for single environment. Set --num_envs to 1.")

    # Load environment and experiment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Load algorithm-specific config, fallback to generic config
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # Determine checkpoint path
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Pretrained checkpoint unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_torch",
            other_dirs=["checkpoints"],
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # Create environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Convert multi-agent to single-agent for PPO
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)

    # Get environment timestep for real-time sync
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # Setup logging if requested
    logger = None
    if args_cli.log:
        logger = CSVLogger(log_dir)

    # Setup video recording wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Configure and instantiate runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    # Load trained policy
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # Execute inference loop
    obs, _ = env.reset()
    timestep = 0
    num_episode = 0

    while simulation_app.is_running():
        start_time = time.time()

        # Forward pass in evaluation mode
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)

            # Extract deterministic actions
            if hasattr(env, "possible_agents"):
                # Multi-agent case
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                # Single-agent case
                actions = outputs[-1].get("mean_actions", outputs[0])

            # Step environment
            obs, rew, terminated, truncated, info = env.step(actions)

        # Exit
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # Real-time synchronization
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Log metrics
        if logger:
            if terminated or truncated:
                num_episode += 1
                logger.save()
                if num_episode >= args_cli.log:
                    break
            logger.log(info["metrics"])

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
