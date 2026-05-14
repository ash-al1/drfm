# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali

import json
import logging
import os
import pickle
import re
import time
from argparse import Namespace
from collections import deque
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper

log = logging.getLogger(__name__)


class EpisodeStatsWrapper(gym.Wrapper):
    """Tracks per-episode returns/lengths and saves best checkpoints."""

    def __init__(
        self,
        env: gym.Env,
        print_every: int = 1000,
        run_dir: str | None = None,
        agent_ref: list | None = None,
        total_timesteps: int | None = None,
    ) -> None:
        super().__init__(env)
        self._print_every = print_every
        self._run_dir = run_dir
        self._agent_ref = agent_ref
        self._total_timesteps = total_timesteps
        self._best_return = -float("inf")
        self._best_step = 0
        self._step = 0
        self._ep_returns: deque = deque(maxlen=200)
        self._ep_lengths: deque = deque(maxlen=200)
        self._ep_timeouts: deque = deque(maxlen=200)
        self._current_returns: torch.Tensor | None = None
        self._current_lengths: torch.Tensor | None = None
        self._t_start = time.time()
        self._t_last = time.time()
        self._step_last = 0
        # lock tracking over the current print_every window
        self._lock_steps_window: float = 0.0
        self._window_steps: int = 0
        self._tech_counts: list[float] = [0.0, 0.0, 0.0, 0.0]  # OFF, RGPO, VGPO, RVGPO
        self._illum_window: float = 0.0
        self._reward_accum: dict[str, float] = {}
        self._term_counts: dict[str, int] = {}

    def step(self, action: Any) -> tuple:
        """Step the environment and accumulate episode statistics."""
        obs, rew, terminated, truncated, info = super().step(action)
        self._step += 1
        self._window_steps += 1

        rl_env = self.env
        while hasattr(rl_env, "env"):
            rl_env = rl_env.env
        if hasattr(rl_env, "reward_manager"):
            rm = rl_env.reward_manager
            for i, name in enumerate(rm._term_names):
                val = rm._step_reward[:, i].mean().item()
                self._reward_accum[name] = self._reward_accum.get(name, 0.0) + val
        if hasattr(rl_env, "termination_manager"):
            tm = rl_env.termination_manager
            for i, name in enumerate(tm._term_names):
                count = int(tm._term_dones[:, i].sum().item())
                self._term_counts[name] = self._term_counts.get(name, 0) + count

        if self._current_returns is None:
            self._current_returns = torch.zeros_like(rew)
            self._current_lengths = torch.zeros(rew.shape[0], dtype=torch.int32, device=rew.device)

        self._current_returns += rew
        self._current_lengths += 1

        m = info.get("metrics", {})
        any_lock = m.get("any_lock")
        if any_lock is not None:
            self._lock_steps_window += any_lock.float().mean().item()

        technique = m.get("drfm_technique")
        if technique is not None:
            for t in range(4):
                self._tech_counts[t] += (technique == t).float().mean().item()

        illum = m.get("illumination_penalty")
        if illum is not None:
            self._illum_window += illum.float().mean().item()

        done = terminated | truncated
        if done.any():
            done_idx = done.nonzero(as_tuple=False).squeeze(-1)
            for i in done_idx:
                self._ep_returns.append(self._current_returns[i].item())
                self._ep_lengths.append(self._current_lengths[i].item())
                self._ep_timeouts.append(bool(truncated[i].item()))
            self._current_returns[done_idx] = 0.0
            self._current_lengths[done_idx] = 0

        if self._step % self._print_every == 0 and self._ep_returns:
            self._log_progress(info)

        return obs, rew, terminated, truncated, info

    def _log_progress(self, info: dict) -> None:
        """Emit a single progress line, write TB scalars, and save checkpoint if return improved."""
        now = time.time()
        fps = (self._step - self._step_last) / max(now - self._t_last, 1e-6)
        elapsed = now - self._t_start
        self._t_last = now
        self._step_last = self._step

        episode_returns = list(self._ep_returns)
        mean_r = sum(episode_returns) / len(episode_returns)
        std_r = (sum((r - mean_r) ** 2 for r in episode_returns) / len(episode_returns)) ** 0.5
        timeout_rate = sum(self._ep_timeouts) / len(self._ep_timeouts)
        mean_ep_len = sum(self._ep_lengths) / len(self._ep_lengths)
        window_steps = self._window_steps
        lock_rate = self._lock_steps_window / max(window_steps, 1)
        tech_total = sum(self._tech_counts) or 1.0
        tech_fracs = [c / tech_total for c in self._tech_counts]
        illum_mean = self._illum_window / max(window_steps, 1)

        self._lock_steps_window = 0.0
        self._window_steps = 0
        self._tech_counts = [0.0, 0.0, 0.0, 0.0]
        self._illum_window = 0.0
        reward_accum = self._reward_accum
        term_counts = self._term_counts
        self._reward_accum = {}
        self._term_counts = {}

        if self._total_timesteps:
            pct = 100.0 * self._step / self._total_timesteps
            progress = f"{self._step:,}/{self._total_timesteps:,} ({pct:.0f}%)"
        else:
            progress = f"step {self._step:,}"

        elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"
        best_str = f"{self._best_return:+.1f}" if self._best_return > -float("inf") else "n/a"

        _tech_names = ["off", "rgpo", "vgpo", "rvgpo"]
        tech_str = "  ".join(f"{n}:{f:.0%}" for n, f in zip(_tech_names, tech_fracs))
        print(
            f"{progress}  |  {fps:,.0f} fps  |  ret {mean_r:+.1f}±{std_r:.1f}"
            f"  |  best {best_str}  |  ep_len {mean_ep_len:.0f}"
            f"  |  lock {lock_rate:.0%}  timeout {timeout_rate:.0%}"
            f"  |  illum {illum_mean:.3f}"
            f"  |  tech [{tech_str}]"
            f"  |  {elapsed_str}",
            flush=True,
        )

        # Write windowed episode stats to env.extras["metrics"] so skrl logs them to TensorBoard.
        metrics = info.setdefault("metrics", {})
        metrics["train/mean_return"] = torch.tensor(mean_r)
        metrics["train/std_return"] = torch.tensor(std_r)
        metrics["train/lock_rate"] = torch.tensor(lock_rate)
        metrics["train/timeout_rate"] = torch.tensor(timeout_rate)
        metrics["train/mean_ep_length"] = torch.tensor(mean_ep_len)
        metrics["train/illumination_penalty"] = torch.tensor(illum_mean)
        for name, frac in zip(_tech_names, tech_fracs):
            metrics[f"train/tech_{name}"] = torch.tensor(frac)

        if reward_accum:
            steps = max(window_steps, 1)
            parts = sorted(reward_accum.items(), key=lambda x: abs(x[1]), reverse=True)
            rew_str = "  ".join(f"{k}:{v/steps:+.2f}" for k, v in parts)
            print(f"  rewards/step: {rew_str}", flush=True)

        if term_counts:
            term_str = "  ".join(f"{k}:{v}" for k, v in term_counts.items() if v > 0)
            if term_str:
                print(f"  terminations: {term_str}", flush=True)

        if self._run_dir and self._agent_ref and self._agent_ref[0] is not None and mean_r > self._best_return:
            self._best_return = mean_r
            self._best_step = self._step
            agent = self._agent_ref[0]
            torch.save(agent.policy.state_dict(), os.path.join(self._run_dir, "actor.pt"))
            if hasattr(agent, "value"):
                torch.save(agent.value.state_dict(), os.path.join(self._run_dir, "critic.pt"))
            agent.save(os.path.join(self._run_dir, "agent_best.pt"))
            print(f"  new best  {mean_r:+.1f}  @ step {self._step:,}  -- checkpoint saved", flush=True)


def task_slug(task: str) -> str:
    """Convert a task name like 'Isaac-Drone-Recon-v0' to 'drone_recon'."""
    return re.sub(r"-v\d+$", "", task.lower().replace("isaac-drone-", "")).replace("-", "_")


def setup_directories(args: Namespace, agent_cfg: dict, algorithm: str) -> tuple[str, str]:
    """Create output and checkpoint directories; return (log_dir, run_dir)."""
    log_root_path = os.path.abspath(os.path.join("outputs", agent_cfg["agent"]["experiment"]["directory"]))

    log_dir = f"{algorithm}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"

    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    run_dir = os.path.join("models", "checkpoints", f"{algorithm}_{task_slug(args.task)}")
    os.makedirs(run_dir, exist_ok=True)

    return log_dir, run_dir


def create_env(
    args: Namespace,
    env_cfg: Any,
    agent_cfg: dict,
    log_dir: str,
    run_dir: str,
    algorithm: str,
) -> tuple[gym.Env, "EpisodeStatsWrapper", list]:
    """Build and wrap the Isaac env; return (skrl_env, stats_wrapper, agent_ref)."""
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)

    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)

    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    _agent_ref: list = [None]
    stats_wrapper = EpisodeStatsWrapper(
        env,
        print_every=1000,
        run_dir=run_dir,
        agent_ref=_agent_ref,
        total_timesteps=agent_cfg["trainer"]["timesteps"],
    )

    # Isaac exposes an unbounded action space; SAC squashes via tanh so we clamp
    # the declared space to [-1, 1] to keep it consistent with policy output.
    _act_dim = env.unwrapped.action_space.shape[-1]
    env.unwrapped.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(_act_dim,), dtype=np.float32)
    env.unwrapped.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(_act_dim,), dtype=np.float32)

    env = SkrlVecEnvWrapper(stats_wrapper, ml_framework="torch")
    return env, stats_wrapper, _agent_ref


def save_hyperparams(
    env: Any,
    env_cfg: Any,
    agent_cfg: dict,
    args: Namespace,
    run_dir: str,
    algorithm: str,
) -> None:
    """Dump hyperparameters to config.yaml in run_dir."""
    m_cfg = agent_cfg["models"]["policy"]["network"][0]
    a_cfg = agent_cfg["agent"]

    if algorithm == "sac":
        algo_hp: dict = {
            "actor_learning_rate": a_cfg["actor_learning_rate"],
            "critic_learning_rate": a_cfg["critic_learning_rate"],
            "entropy_learning_rate": a_cfg["entropy_learning_rate"],
            "batch_size": a_cfg["batch_size"],
            "memory_size": a_cfg["memory_size"],
            "polyak": a_cfg.get("polyak", 0.005),
        }
    else:
        algo_hp = {
            "learning_rate": a_cfg["learning_rate"],
            "clip_ratio": a_cfg["ratio_clip"],
            "entropy_coef": a_cfg["entropy_loss_scale"],
            "gae_lambda": a_cfg["lambda"],
            "mini_batches": a_cfg["mini_batches"],
            "epochs": a_cfg["learning_epochs"],
            "rollout_steps": a_cfg["rollouts"],
        }

    hp: dict = {
        "algorithm": args.algorithm.upper(),
        "task": task_slug(args.task),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "architecture": "mlp_actor_critic",
        "observation_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "hidden_sizes": list(m_cfg["layers"]),
        "activation": m_cfg["activations"],
        "gamma": a_cfg["discount_factor"],
        "max_grad_norm": a_cfg["grad_norm_clip"],
        "num_envs": env_cfg.scene.num_envs,
        "total_timesteps": agent_cfg["trainer"]["timesteps"],
        **algo_hp,
        "terrain": "flat",
        "obstacles": True,
        "radars": True,
    }
    try:
        hp["waypoints_per_episode"] = env_cfg.commands.target.waypoints_per_episode
    except AttributeError:
        pass
    for attr, key in [
        ("progress", "w_progress"),
        ("heading", "w_heading"),
        ("waypoint_reached", "w_waypoint_reached"),
        ("completion_bonus", "w_completion_bonus"),
        ("terminating", "w_terminating"),
        ("step_penalty", "w_step_penalty"),
        ("alive", "w_alive"),
        ("proximity", "w_proximity"),
        ("forward_speed", "w_forward_speed"),
        ("distance_penalty", "w_distance_penalty"),
        ("ang_vel_l2", "w_ang_vel_l2"),
        ("ang_vel", "w_ang_vel"),
        ("altitude", "w_altitude"),
        ("upright", "w_upright"),
    ]:
        try:
            hp[key] = getattr(env_cfg.rewards, attr).weight
        except AttributeError:
            pass

    yaml_str = yaml.dump(hp, default_flow_style=False, sort_keys=False)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        f.write(yaml_str)


def save_final_checkpoint(agent: Any, stats_wrapper: EpisodeStatsWrapper, run_dir: str, t0: float, agent_cfg: dict) -> None:
    """Save final model weights, skrl checkpoint, training metrics JSON, and log completion summary."""
    torch.save(agent.policy.state_dict(), os.path.join(run_dir, "actor_final.pt"))
    if hasattr(agent, "value"):
        torch.save(agent.value.state_dict(), os.path.join(run_dir, "critic_final.pt"))
    agent.save(os.path.join(run_dir, "agent_final.pt"))

    wall = int(time.time() - t0)
    ep_rets = [r for r in stats_wrapper._ep_returns if isinstance(r, (int, float)) and r == r]
    final_return = sum(ep_rets) / len(ep_rets) if ep_rets else None
    best = stats_wrapper._best_return if stats_wrapper._best_return > -float("inf") else None

    metrics = {
        "best_episode_return": best,
        "best_episode_return_step": stats_wrapper._best_step if best is not None else None,
        "final_episode_return": final_return,
        "success_rate": None,
        "total_training_steps": agent_cfg["trainer"]["timesteps"],
        "wall_time_seconds": wall,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    final_str = f"{final_return:+.1f}" if final_return is not None else "n/a"
    best_str  = f"{best:+.1f}" if best is not None else "n/a"
    step_str  = f"{stats_wrapper._best_step:,}" if best is not None else "n/a"
    print(
        f"done  wall={wall // 60}m{wall % 60:02d}s  final_ret={final_str}"
        f"  best={best_str} @ step {step_str}"
        f"  checkpoint={os.path.join(run_dir, 'agent_final.pt')}",
        flush=True,
    )
