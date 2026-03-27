#
#
# Usage: python3 scripts/train.py --task Isaac-Drone-Recon-v0 --headless --num_envs 4096

import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
from collections import deque
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym

parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC"])
parser.add_argument("--phase", type=int, required=True)
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="drone-recon")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import drfm.envs.isaac  # noqa: F401

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from models.architectures.mlp_actor_critic import MLPActor, MLPCritic, MLPSACCritic
from models.replay_buffers import ReplayBuffer


class LoggedPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_metrics = {}
        self._update_count = 0

    def init(self, trainer_cfg=None):
        super().init(trainer_cfg)
        if self.memory is not None:
            self._current_log_prob = torch.zeros(self.memory.num_envs, 1, device=self.device)

    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        if timestep == 0 and torch.isnan(actions).any():
            print(f"[ERROR] NaN actions at step 0! This should not happen with bounded action space.")
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

    def _update(self, timestep, timesteps):
        self._update_count += 1
        PPO._update(self, timestep, timesteps)
        self.last_metrics = {k: list(v) for k, v in self.tracking_data.items()}


class LoggedSAC(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_metrics = {}
        self._update_count = 0

    def _update(self, timestep, timesteps):
        self._update_count += 1
        SAC._update(self, timestep, timesteps)
        self.last_metrics = {k: list(v) for k, v in self.tracking_data.items()}


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def plot_training_curves(log_dir: str) -> None:
    import numpy as np
    import pandas as pd
    matplotlib.use("Agg")
    csv_files = glob.glob(os.path.join(log_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
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


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env, print_every=1000, run_dir=None, agent_ref=None, total_timesteps=None):
        super().__init__(env)
        self._print_every = print_every
        self._run_dir = run_dir
        self._agent_ref = agent_ref
        self._total_timesteps = total_timesteps
        self._best_return = -float("inf")
        self._best_step = 0
        self._step = 0
        self._ep_returns = deque(maxlen=200)
        self._ep_lengths = deque(maxlen=200)
        self._ep_timeouts = deque(maxlen=200)
        self._current_returns = None
        self._current_lengths = None
        self._t_start = time.time()
        self._t_last = time.time()
        self._step_last = 0

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        self._step += 1

        if self._current_returns is None:
            self._current_returns = torch.zeros_like(rew)
            self._current_lengths = torch.zeros(rew.shape[0], dtype=torch.int32, device=rew.device)

        self._current_returns += rew
        self._current_lengths += 1

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
            now = time.time()
            elapsed = now - self._t_start
            fps = (self._step - self._step_last) / max(now - self._t_last, 1e-6)
            self._t_last = now
            self._step_last = self._step

            rets = list(self._ep_returns)
            n = len(rets)
            mean_r = sum(rets) / n
            std_r = (sum((r - mean_r) ** 2 for r in rets) / n) ** 0.5
            mean_len = sum(self._ep_lengths) / len(self._ep_lengths)
            timeout_pct = 100.0 * sum(self._ep_timeouts) / len(self._ep_timeouts)

            if self._total_timesteps:
                pct = 100.0 * self._step / self._total_timesteps
                eta = (self._total_timesteps - self._step) / max(fps, 1)
                progress = f"{self._step:,} / {self._total_timesteps:,}  ({pct:.1f}%)"
                time_str = f"{_fmt_time(elapsed)} elapsed  eta {_fmt_time(eta)}"
            else:
                progress = f"step {self._step:,}"
                time_str = f"{_fmt_time(elapsed)} elapsed"

            print(
                f"\n-- train  {progress}  {fps:,.0f} fps  {time_str}\n"
                f"   return  {mean_r:+.1f} +/- {std_r:.1f}  "
                f"(min {min(rets):.0f}  max {max(rets):.0f})  "
                f"n={n}  len={mean_len:.0f}  timeout={timeout_pct:.0f}%"
            )

            if self._agent_ref and self._agent_ref[0] is not None:
                m = self._agent_ref[0].last_metrics
                updates = self._agent_ref[0]._update_count
                if m:
                    def _avg(key):
                        vals = m.get(key, [])
                        return sum(vals) / len(vals) if vals else 0.0

                    losses = {k.split("Loss / ")[-1]: _avg(k) for k in m if k.startswith("Loss / ")}
                    lr_key = [k for k in m if "Learning rate" in k]
                    lr = _avg(lr_key[0]) if lr_key else 0.0
                    loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
                    print(f"   losses  {loss_str}  lr={lr:.2e}  updates={updates}")

            if self._run_dir and self._agent_ref and self._agent_ref[0] is not None and mean_r > self._best_return:
                self._best_return = mean_r
                self._best_step = self._step
                agent = self._agent_ref[0]
                torch.save(agent.policy.state_dict(), os.path.join(self._run_dir, "actor.pt"))
                if hasattr(agent, "value"):
                    torch.save(agent.value.state_dict(), os.path.join(self._run_dir, "critic.pt"))
                agent.save(os.path.join(self._run_dir, "agent_best.pt"))
                print(f"   >>> NEW BEST  {mean_r:+.1f}  @ step {self._step:,}  - checkpoint saved")

        return obs, rew, terminated, truncated, info


def _task_slug(task: str, phase: int | None) -> str:
    import re
    slug = re.sub(r"-v\d+$", "", task.lower().replace("isaac-drone-", "")).replace("-", "_")
    return f"{slug}_phase{phase}" if phase is not None else slug


def _build_agent(env, agent_cfg):
    a = agent_cfg["agent"]
    m = agent_cfg["models"]
    hidden_sizes = tuple(m["policy"]["network"][0]["layers"])
    activation = m["policy"]["network"][0]["activations"]

    models = {
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

    memory = RandomMemory(memory_size=a["rollouts"], num_envs=env.num_envs, device=env.device)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg.update({
        "rollouts":                       a["rollouts"],
        "learning_epochs":                a["learning_epochs"],
        "mini_batches":                   a["mini_batches"],
        "discount_factor":                a["discount_factor"],
        "lambda":                         a["lambda"],
        "learning_rate":                  a["learning_rate"],
        "learning_rate_scheduler":        KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": a.get("learning_rate_scheduler_kwargs", {}),
        "state_preprocessor":             RunningStandardScaler,
        "state_preprocessor_kwargs":      {"size": env.observation_space, "device": env.device},
        "value_preprocessor":             RunningStandardScaler,
        "value_preprocessor_kwargs":      {"size": 1, "device": env.device},
        "random_timesteps":               a.get("random_timesteps", 0),
        "learning_starts":                a.get("learning_starts", 0),
        "grad_norm_clip":                 a.get("grad_norm_clip", 1.0),
        "ratio_clip":                     a.get("ratio_clip", 0.2),
        "value_clip":                     a.get("value_clip", 0.2),
        "clip_predicted_values":          a.get("clip_predicted_values", True),
        "entropy_loss_scale":             a.get("entropy_loss_scale", 0.005),
        "value_loss_scale":               a.get("value_loss_scale", 1.0),
        "kl_threshold":                   a.get("kl_threshold", 0.0),
        "time_limit_bootstrap":           False,
        "experiment": {
            "directory":           a["experiment"]["directory"],
            "experiment_name":     a["experiment"]["experiment_name"],
            "write_interval":      1000,
            "checkpoint_interval": a["experiment"].get("checkpoint_interval", 0),
        },
    })
    if a.get("rewards_shaper_scale") is not None:
        scale = a["rewards_shaper_scale"]
        cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * scale

    return LoggedPPO(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


def _build_sac_agent(env, agent_cfg):
    a = agent_cfg["agent"]
    m = agent_cfg["models"]
    hidden_sizes = tuple(m["policy"]["network"][0]["layers"])
    activation = m["policy"]["network"][0]["activations"]

    critic_kwargs = dict(
        hidden_sizes=hidden_sizes, activation=activation,
        clip_actions=m["critic"].get("clip_actions", False),
    )
    models = {
        "policy": MLPActor(
            env.observation_space, env.action_space, env.device,
            hidden_sizes=hidden_sizes, activation=activation,
            clip_actions=m["policy"].get("clip_actions", False),
            clip_log_std=m["policy"].get("clip_log_std", True),
            min_log_std=m["policy"].get("min_log_std", -20.0),
            max_log_std=m["policy"].get("max_log_std", 2.0),
        ),
        "critic_1":        MLPSACCritic(env.observation_space, env.action_space, env.device, **critic_kwargs),
        "critic_2":        MLPSACCritic(env.observation_space, env.action_space, env.device, **critic_kwargs),
        "target_critic_1": MLPSACCritic(env.observation_space, env.action_space, env.device, **critic_kwargs),
        "target_critic_2": MLPSACCritic(env.observation_space, env.action_space, env.device, **critic_kwargs),
    }

    memory = ReplayBuffer(capacity=a["memory_size"], num_envs=env.num_envs, device=env.device)

    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg.update({
        "gradient_steps":          a.get("gradient_steps", 1),
        "batch_size":              a["batch_size"],
        "discount_factor":         a["discount_factor"],
        "polyak":                  a.get("polyak", 0.005),
        "actor_learning_rate":     a["actor_learning_rate"],
        "critic_learning_rate":    a["critic_learning_rate"],
        "entropy_learning_rate":   a["entropy_learning_rate"],
        "initial_entropy_value":   a.get("initial_entropy_value", 0.2),
        "target_entropy":          a.get("target_entropy", None),
        "learn_entropy":           a.get("learn_entropy", True),
        "state_preprocessor":      RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "random_timesteps":        a.get("random_timesteps", 0),
        "learning_starts":         a.get("learning_starts", 0),
        "grad_norm_clip":          a.get("grad_norm_clip", 0),
        "experiment": {
            "directory":           a["experiment"]["directory"],
            "experiment_name":     a["experiment"]["experiment_name"],
            "write_interval":      1000,
            "checkpoint_interval": a["experiment"].get("checkpoint_interval", 0),
        },
    })

    return LoggedSAC(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.max_iterations is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]

    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]

    log_root_path = os.path.abspath(
        os.path.join("outputs", f"phase_{args_cli.phase}", agent_cfg["agent"]["experiment"]["directory"])
    )
    print(f"[INFO] Logging to: {log_root_path}")
    print(f"[INFO] TensorBoard: tensorboard --logdir {log_root_path}")

    log_dir = f"{algorithm}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"

    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    if args_cli.wandb:
        agent_cfg["agent"]["experiment"]["wandb"] = True
        agent_cfg["agent"]["experiment"]["wandb_kwargs"] = {
            "project": args_cli.wandb_project,
            "name": os.path.basename(log_dir),
        }

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)

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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _task_slug(args_cli.task, args_cli.phase)
    run_name = f"{algorithm}_{slug}_{ts}"
    run_dir = os.path.join("models", "checkpoints", run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join("models", "configs"), exist_ok=True)

    _agent_ref = [None]
    stats_wrapper = EpisodeStatsWrapper(
        env, print_every=1000, run_dir=run_dir, agent_ref=_agent_ref,
        total_timesteps=agent_cfg["trainer"]["timesteps"],
    )

    import gymnasium as _gym
    import numpy as _np
    _act_dim = env.unwrapped.action_space.shape[-1]
    env.unwrapped.action_space = _gym.spaces.Box(low=-1.0, high=1.0, shape=(_act_dim,), dtype=_np.float32)
    env.unwrapped.single_action_space = _gym.spaces.Box(low=-1.0, high=1.0, shape=(_act_dim,), dtype=_np.float32)

    env = SkrlVecEnvWrapper(stats_wrapper, ml_framework="torch")

    m_cfg = agent_cfg["models"]["policy"]["network"][0]
    a_cfg = agent_cfg["agent"]
    if algorithm == "sac":
        algo_hp = {
            "actor_learning_rate":   a_cfg["actor_learning_rate"],
            "critic_learning_rate":  a_cfg["critic_learning_rate"],
            "entropy_learning_rate": a_cfg["entropy_learning_rate"],
            "batch_size":            a_cfg["batch_size"],
            "memory_size":           a_cfg["memory_size"],
            "polyak":                a_cfg.get("polyak", 0.005),
        }
    else:
        algo_hp = {
            "learning_rate": a_cfg["learning_rate"],
            "clip_ratio":    a_cfg["ratio_clip"],
            "entropy_coef":  a_cfg["entropy_loss_scale"],
            "gae_lambda":    a_cfg["lambda"],
            "mini_batches":  a_cfg["mini_batches"],
            "epochs":        a_cfg["learning_epochs"],
            "rollout_steps": a_cfg["rollouts"],
        }
    hp = {
        "algorithm":       args_cli.algorithm.upper(),
        "task":            slug,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "architecture":    "mlp_actor_critic",
        "observation_dim": env.observation_space.shape[0],
        "action_dim":      env.action_space.shape[0],
        "hidden_sizes":    list(m_cfg["layers"]),
        "activation":      m_cfg["activations"],
        "gamma":           a_cfg["discount_factor"],
        "max_grad_norm":   a_cfg["grad_norm_clip"],
        "num_envs":        env_cfg.scene.num_envs,
        "total_timesteps": agent_cfg["trainer"]["timesteps"],
        **algo_hp,
        "waypoints_per_episode": env_cfg.commands.target.waypoints_per_episode,
        "w_progress":         env_cfg.rewards.progress.weight,
        "w_heading":          env_cfg.rewards.heading.weight,
        "w_arrived":          env_cfg.rewards.arrived.weight,
        "w_completion_bonus": env_cfg.rewards.completion_bonus.weight,
        "w_terminating":      env_cfg.rewards.terminating.weight,
        "w_step_penalty":     env_cfg.rewards.step_penalty.weight,
        "w_proximity":        env_cfg.rewards.proximity.weight,
        "terrain":         "flat",
        "obstacles":       True,
        "radars":          args_cli.phase >= 2,
    }
    try:
        hp["w_forward_speed"] = env_cfg.rewards.forward_speed.weight
    except AttributeError:
        pass
    try:
        hp["w_distance_penalty"] = env_cfg.rewards.distance_penalty.weight
    except AttributeError:
        pass
    try:
        hp["w_ang_vel_l2"] = env_cfg.rewards.ang_vel_l2.weight
    except AttributeError:
        pass
    yaml_str = yaml.dump(hp, default_flow_style=False, sort_keys=False)
    for dest in [os.path.join(run_dir, "config.yaml"), os.path.join("models", "configs", f"{run_name}.yaml")]:
        with open(dest, "w") as f:
            f.write(yaml_str)
    print(f"[INFO] Config written to: {run_dir}/config.yaml")

    if algorithm == "sac":
        agent = _build_sac_agent(env, agent_cfg)
    else:
        agent = _build_agent(env, agent_cfg)
    _agent_ref[0] = agent

    print(f"[INFO] Policy params: {sum(p.numel() for p in agent.policy.parameters()):,}")
    if hasattr(agent, "value"):
        print(f"[INFO] Value  params: {sum(p.numel() for p in agent.value.parameters()):,}")
    print(f"[INFO] obs_space: {env.observation_space}")
    print(f"[INFO] act_space: {env.action_space}")

    if resume_path:
        print(f"[INFO] Resuming from: {resume_path}")
        agent.load(resume_path)

    t0 = time.time()
    SequentialTrainer(
        cfg={
            "timesteps": agent_cfg["trainer"]["timesteps"],
            "environment_info": agent_cfg["trainer"].get("environment_info", "log"),
            "close_environment_at_exit": False,
        },
        env=env,
        agents=agent,
    ).train()

    torch.save(agent.policy.state_dict(), os.path.join(run_dir, "actor_final.pt"))
    if hasattr(agent, "value"):
        torch.save(agent.value.state_dict(), os.path.join(run_dir, "critic_final.pt"))
    agent.save(os.path.join(run_dir, "agent_final.pt"))

    ep_rets = [r for r in stats_wrapper._ep_returns if isinstance(r, (int, float)) and r == r]
    metrics = {
        "best_episode_return":      stats_wrapper._best_return if stats_wrapper._best_return > -float("inf") else None,
        "best_episode_return_step": stats_wrapper._best_step if stats_wrapper._best_return > -float("inf") else None,
        "final_episode_return":     sum(ep_rets) / len(ep_rets) if ep_rets else None,
        "success_rate":             None,
        "total_training_steps":     agent_cfg["trainer"]["timesteps"],
        "wall_time_seconds":        int(time.time() - t0),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Artifacts saved to: {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
