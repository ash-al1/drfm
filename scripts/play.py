# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Usage:
#   python3 scripts/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1
#   python3 scripts/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1 --checkpoint path/to/model.pt

import argparse
import glob
import os
import sys
import time

import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run inference for a trained drone agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--renderer", type=str, default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"])
parser.add_argument("--log", type=int, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import drfm.envs.isaac  # noqa: F401
from utils.logger import CSVLogger

from skrl.utils.runner.torch import Runner
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from models.architectures.mlp_actor_critic import MLPActor, MLPCritic

algorithm = args_cli.algorithm.lower()


def main() -> None:
    if args_cli.log and args_cli.num_envs > 1:
        raise ValueError("Logging requires --num_envs 1.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    log_root_path = os.path.abspath(
        os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    )
    print(f"[INFO] Loading experiment from: {log_root_path}")

    resume_path = None
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] No pretrained checkpoint available.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        try:
            resume_path = get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}", other_dirs=["checkpoints"])
        except (ValueError, FileNotFoundError):
            print("[INFO] No checkpoint found — running with untrained policy.")

    log_dir = os.path.dirname(os.path.dirname(resume_path)) if resume_path else log_root_path

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    logger = CSVLogger(log_dir) if args_cli.log else None

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

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    if "Recon" in args_cli.task:
        if not args_cli.checkpoint and not args_cli.use_pretrained_checkpoint:
            resume_path = None

        a = experiment_cfg["agent"]
        m = experiment_cfg["models"]
        hidden_sizes = tuple(m["policy"]["network"][0]["layers"])
        activation = m["policy"]["network"][0]["activations"]

        models_dict = {
            "policy": MLPActor(
                env.observation_space, env.action_space, env.device,
                hidden_sizes=hidden_sizes, activation=activation,
                clip_actions=m["policy"].get("clip_actions", False),
                clip_log_std=m["policy"].get("clip_log_std", True),
                min_log_std=m["policy"].get("min_log_std", -20.0),
                max_log_std=m["policy"].get("max_log_std", 2.0),
            ),
            "value": MLPCritic(
                env.observation_space, env.action_space, env.device,
                hidden_sizes=hidden_sizes, activation=activation,
                clip_actions=m["value"].get("clip_actions", False),
            ),
        }
        memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=env.device)
        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg.update({
            "state_preprocessor":        RunningStandardScaler,
            "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
            "value_preprocessor":        RunningStandardScaler,
            "value_preprocessor_kwargs": {"size": 1, "device": env.device},
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        })
        agent = PPO(
            models=models_dict, memory=memory, cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
        )

        if resume_path is None:
            for pattern in ("agent_final.pt", "agent_best.pt"):
                candidates = sorted(glob.glob(os.path.join("models", "checkpoints", "*", pattern)))
                if candidates:
                    resume_path = candidates[-1]
                    print(f"[INFO] Auto-discovered checkpoint: {resume_path}")
                    break
            else:
                print("[INFO] No checkpoint found — running with untrained policy.")

        if resume_path:
            print(f"[INFO] Loading checkpoint: {resume_path}")
            agent.load(resume_path)
        agent.set_running_mode("eval")

    else:
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, experiment_cfg)
        if resume_path:
            print(f"[INFO] Loading checkpoint: {resume_path}")
            runner.agent.load(resume_path)
        runner.agent.set_running_mode("eval")
        agent = runner.agent

    obs, _ = env.reset()
    timestep = 0
    num_episode = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rew, terminated, truncated, info = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

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
