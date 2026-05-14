# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali
#
# Copyright (c) 2025, Kousheek Chakraborty
# Original work licensed under the BSD-3-Clause License.
# Built on the IsaacLab framework (https://github.com/isaac-sim/IsaacLab).

import argparse
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="singleDRFM")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "PPO_GRU", "SAC", "MAPPO"])
parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

logging.basicConfig(level=getattr(logging, args_cli.log_level), format="%(message)s", force=True)

import torch  # noqa: E402 

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import drfm.isaac  # noqa: F401

from skrl.trainers.torch import SequentialTrainer
from drfm.agents.builder import build_mappo_agent, build_ppo_agent, build_ppo_gru_agent, build_sac_agent
from drfm.agents.train_utils import (
    EpisodeStatsWrapper,  # noqa: F401
    create_env,
    save_final_checkpoint,
    save_hyperparams,
    setup_directories,
)

algorithm = args_cli.algorithm.lower()
if algorithm == "ppo_gru":
    agent_cfg_entry_point = "skrl_ppo_gru_cfg_entry_point"
elif algorithm == "ppo":
    agent_cfg_entry_point = "skrl_cfg_entry_point"
elif algorithm == "mappo":
    agent_cfg_entry_point = "skrl_mappo_cfg_entry_point"
else:
    agent_cfg_entry_point = f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    """Entry point: wire together setup, env creation, agent training, and checkpointing."""
    logging.basicConfig(level=getattr(logging, args_cli.log_level), format="%(message)s", force=True)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.max_iterations is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]

    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]

    log_dir, run_dir = setup_directories(args_cli, agent_cfg, algorithm)
    env, stats_wrapper, _agent_ref = create_env(args_cli, env_cfg, agent_cfg, log_dir, run_dir, algorithm)

    if algorithm == "ppo_gru":
        agent = build_ppo_gru_agent(env, agent_cfg)
    elif algorithm == "sac":
        agent = build_sac_agent(env, agent_cfg)
    elif algorithm == "mappo":
        agent = build_mappo_agent(env, agent_cfg)
    else:
        agent = build_ppo_agent(env, agent_cfg)
    _agent_ref[0] = agent

    # Policy may be per-agent dict for MAPPO; use the first agent's policy for param count
    policy_ref = agent.policy if hasattr(agent, "policy") else next(iter(agent.policies.values()))
    param_count = sum(p.numel() for p in policy_ref.parameters())
    print(
        f"algorithm={args_cli.algorithm.upper()}  task={args_cli.task}"
        f"  num_envs={env_cfg.scene.num_envs}  params={param_count:,}  device={agent.device}\n"
        f"run_dir={run_dir}\ntotal_timesteps={agent_cfg['trainer']['timesteps']:,}",
        flush=True,
    )

    if algorithm != "mappo":
        robot = env.unwrapped.scene["robot"]
        print("Body names:", robot.body_names)
        print("Per-body masses:", robot.data.default_mass[0])

    if algorithm != "mappo":
        save_hyperparams(env, env_cfg, agent_cfg, args_cli, run_dir, algorithm)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    if resume_path:
        agent.load(resume_path)

    t0 = time.time()
    SequentialTrainer(
        cfg={
            "timesteps": agent_cfg["trainer"]["timesteps"],
            "environment_info": agent_cfg["trainer"].get("environment_info", "episode"),
            "close_environment_at_exit": False,
        },
        env=env,
        agents=agent,
    ).train()

    save_final_checkpoint(agent, stats_wrapper, run_dir, t0, agent_cfg)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
