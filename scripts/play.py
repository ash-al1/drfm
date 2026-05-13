# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali
#
# Copyright (c) 2025, Kousheek Chakraborty
# Original work licensed under the BSD-3-Clause License.
# Built on the IsaacLab framework (https://github.com/isaac-sim/IsaacLab).

import argparse
import glob
import logging
import math
import os
import time

import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run inference for a trained drone agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="singleDRFM")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC"])
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--renderer", type=str, default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"])
parser.add_argument("--debug", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import drfm.isaac  # noqa: F401

logging.basicConfig(level=logging.DEBUG if args_cli.debug else logging.INFO, format="%(message)s", force=True)

from drfm.agents.builder import build_ppo_agent, build_sac_agent
from drfm.agents.play_utils import CameraFollower

log = logging.getLogger(__name__)

# Phase-2 obs includes base(14) + 3 radars×10 + drfm_state(8) = 52 dims minimum,
# but the last accessed index is 61 (drfm power), so phase-2 obs dim >= 62.
PHASE2_MIN_OBS_DIM = 62

algorithm = args_cli.algorithm.lower()


def _resolve_checkpoint() -> str | None:
    if args_cli.checkpoint:
        return os.path.abspath(args_cli.checkpoint)
    for pattern in ("agent_best.pt", "agent_final.pt"):
        candidates = sorted(glob.glob(os.path.join("models", "checkpoints", "*", pattern)))
        matching = [c for c in candidates if os.path.basename(os.path.dirname(c)).startswith(algorithm)]
        if matching:
            return matching[-1]
    return None


def _quat_to_euler_deg(q):
    w, x, y, z = q
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr, cosr))
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny, cosy))
    return roll, pitch, yaw


def main() -> None:
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

    resume_path = _resolve_checkpoint()
    log_dir = os.path.dirname(os.path.dirname(resume_path)) if resume_path else "."

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)

    raw_env = env.unwrapped

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        log.info("Recording video")
        print_dict(video_kwargs, nesting=4)  # isaaclab helper, not a bare print
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    if algorithm == "sac":
        agent = build_sac_agent(env, experiment_cfg, training=False)
    else:
        agent = build_ppo_agent(env, experiment_cfg, training=False)

    if resume_path:
        log.info("Loading checkpoint: %s", resume_path)
        agent.load(resume_path)
    else:
        log.warning("No checkpoint found - running with untrained policy.")

    agent.set_running_mode("eval")

    obs, _ = env.reset()
    timestep = 0
    num_episode = 0
    ep_return = 0.0
    ep_steps = 0
    prev_episode_ended = False

    log.info("Observation shape: %s", obs.shape)
    log.info("Episode length: %ss", env_cfg.episode_length_s)

    wp_total = env_cfg.commands.target.waypoints_per_episode
    cam = CameraFollower()

    while simulation_app.is_running():
        start_time = time.time()

        # Save pre-step state - env auto-resets on terminal, wiping this info.
        if args_cli.num_envs == 1:
            _pre_obs     = obs[0].cpu().clone()
            _pre_pos     = raw_env.scene["robot"].data.root_pos_w[0].cpu().clone()
            cam.update(_pre_pos.numpy())

        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rew, terminated, truncated, info = env.step(actions)

        if args_cli.num_envs == 1:
            o = obs[0].cpu()
            r = rew[0].item()
            t = terminated[0].item()
            tr = truncated[0].item()
            ep_return += r
            ep_steps += 1

            if args_cli.debug:
                target_b  = _pre_obs[:3]
                wp_rem    = int(round(_pre_obs[3].item() * wp_total))
                quat      = _pre_obs[4:8]
                lin_vel   = _pre_obs[8:11]
                ang_vel   = _pre_obs[11:14]
                _, _, yaw = _quat_to_euler_deg(quat.tolist())
                dist      = torch.norm(target_b).item()
                spd       = torch.norm(lin_vel).item()
                angv      = torch.norm(ang_vel).item()
                alt       = _pre_pos[2].item()
                phase2    = _pre_obs.shape[0] >= PHASE2_MIN_OBS_DIM

                if torch.isnan(_pre_obs).any().item():
                    log.error("NaN in observations  ep=%d  s=%d", num_episode, ep_steps)

                if ep_steps <= 3 or ep_steps % 100 == 0:
                    log.debug(
                        "ep=%d  s=%4d  alt=%4.1f  dist=%5.1f  wp=%d/%d  "
                        "yaw=%+6.1f  spd=%4.1f  angv=%4.1f  rew=%+7.2f",
                        num_episode, ep_steps, alt, dist, wp_rem, wp_total,
                        yaw, spd, angv, r,
                    )

                    if phase2:
                        _RADAR = ["SAcq", "PD  ", "Mono"]
                        _STATE = ["Search", "Detect", "Track ", "LOCK  "]
                        _TECH  = ["OFF", "RGPO", "VGPO", "RVGPO"]
                        radar_parts = []
                        for i in range(3):
                            b     = 14 + i * 10
                            state = int(_pre_obs[b + 5 : b + 9].argmax().item())
                            tq    = _pre_obs[b + 9].item()
                            radar_parts.append(f"{_RADAR[i]}:{_STATE[state]} {tq:.2f}")
                        tech      = int(_pre_obs[54:58].argmax().item())
                        por       = _pre_obs[58].item() * 500
                        vpor      = _pre_obs[59].item() * 200
                        pwr       = _pre_obs[61].item()
                        param_str = ""
                        if tech == 1:
                            param_str = f"  por={por:.0f}m/s"
                        elif tech == 2:
                            param_str = f"  vpor={vpor:.0f}m/s^2"
                        elif tech == 3:
                            param_str = f"  por={por:.0f}  vpor={vpor:.0f}"
                        pwr_str = f"{pwr*100:.0f}%" if pwr > 0.0 else "DEPLETED"
                        log.debug(
                            "         radar  %s\n         drfm   %s%s  pwr=%s",
                            "  |  ".join(radar_parts), _TECH[tech], param_str, pwr_str,
                        )

            if t or tr:
                # Use pre-step state - post-step is already the next episode.
                wp_done = wp_total - int(round(_pre_obs[3].item() * wp_total))
                outcome = "SUCCESS" if t and wp_done >= wp_total else ("KILLED" if t else "TIMEOUT")
                log.info(
                    "ep=%3d  %s  steps=%4d  return=%+8.2f  wp=%d/%d  pos=(%.1f,%.1f,%.1f)",
                    num_episode, outcome, ep_steps, ep_return,
                    wp_done, wp_total,
                    _pre_pos[0], _pre_pos[1], _pre_pos[2],
                )

        if terminated.any() or truncated.any():
            num_episode += 1
            ep_return = 0.0
            ep_steps = 0

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
