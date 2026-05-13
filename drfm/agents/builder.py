# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali

import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from models.architectures.mlp_actor_critic import MLPActor, MLPCritic, MLPSACCritic
from models.architectures.rnn_actor_critic import GRUActor, GRUCritic


class GRULoggedPPO(PPO):
    """PPO subclass that resets GRU hidden states on episode termination/truncation."""

    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        done = (terminated | truncated).squeeze(-1)
        if done.any():
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.policy.reset_hidden(env_ids)
            self.value.reset_hidden(env_ids)


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


def build_ppo_agent(env, agent_cfg, *, memory_size: int | None = None, training: bool = True):
    """Build a PPO agent.

    Args:
        memory_size: replay buffer size. Defaults to agent_cfg rollouts (training) or 1 (inference).
        training: if True, use full training config and LoggedPPO; if False, use minimal inference config and PPO.
    """
    agent_params = agent_cfg["agent"]
    model_cfg    = agent_cfg["models"]
    hidden_sizes = tuple(model_cfg["policy"]["network"][0]["layers"])
    activation   = model_cfg["policy"]["network"][0]["activations"]

    models = {
        "policy": MLPActor(
            env.observation_space, env.action_space, env.device,
            hidden_sizes=hidden_sizes, activation=activation,
            clip_actions=model_cfg["policy"].get("clip_actions", False),
            clip_log_std=model_cfg["policy"].get("clip_log_std", True),
            min_log_std=model_cfg["policy"].get("min_log_std", -20.0),
            max_log_std=model_cfg["policy"].get("max_log_std", 2.0),
        ),
        "value": MLPCritic(
            env.observation_space, env.action_space, env.device,
            hidden_sizes=hidden_sizes, activation=activation,
            clip_actions=model_cfg["value"].get("clip_actions", False),
        ),
    }

    if memory_size is None:
        memory_size = agent_params["rollouts"] if training else 1

    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg.update({
        "state_preprocessor":        RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "value_preprocessor":        RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": env.device},
    })

    if training:
        cfg.update({
            "rollouts":                       agent_params["rollouts"],
            "learning_epochs":                agent_params["learning_epochs"],
            "mini_batches":                   agent_params["mini_batches"],
            "discount_factor":                agent_params["discount_factor"],
            "lambda":                         agent_params["lambda"],
            "learning_rate":                  agent_params["learning_rate"],
            "learning_rate_scheduler":        KLAdaptiveLR,
            "learning_rate_scheduler_kwargs": agent_params.get("learning_rate_scheduler_kwargs", {}),
            "random_timesteps":               agent_params.get("random_timesteps", 0),
            "learning_starts":                agent_params.get("learning_starts", 0),
            "grad_norm_clip":                 agent_params.get("grad_norm_clip", 1.0),
            "ratio_clip":                     agent_params.get("ratio_clip", 0.2),
            "value_clip":                     agent_params.get("value_clip", 0.2),
            "clip_predicted_values":          agent_params.get("clip_predicted_values", True),
            "entropy_loss_scale":             agent_params.get("entropy_loss_scale", 0.005),
            "value_loss_scale":               agent_params.get("value_loss_scale", 1.0),
            "kl_threshold":                   agent_params.get("kl_threshold", 0.0),
            "time_limit_bootstrap":           False,
            "experiment": {
                "directory":           agent_params["experiment"]["directory"],
                "experiment_name":     agent_params["experiment"]["experiment_name"],
                "write_interval":      1000,
                "checkpoint_interval": agent_params["experiment"].get("checkpoint_interval", 0),
            },
        })
        if agent_params.get("rewards_shaper_scale") is not None:
            scale = agent_params["rewards_shaper_scale"]
            cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * scale
        agent_cls = LoggedPPO
    else:
        cfg.update({"experiment": {"write_interval": 0, "checkpoint_interval": 0}})
        agent_cls = PPO

    return agent_cls(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


def build_ppo_gru_agent(env, agent_cfg, *, memory_size: int | None = None, training: bool = True,
                        temporal_dim: int = 32, gru_hidden: int = 64):
    """Build a PPO agent with GRU actor and critic.

    Args:
        temporal_dim: number of leading obs dims fed into the GRU (default 32 = RWR stream).
        gru_hidden: GRU hidden size.
        memory_size: rollout buffer size. Defaults to agent_cfg rollouts (training) or 1 (inference).
        training: if True, use full training config; if False, use minimal inference config.
    """
    agent_params = agent_cfg["agent"]
    model_cfg    = agent_cfg["models"]
    hidden_sizes = tuple(model_cfg["policy"]["network"][0]["layers"])
    activation   = model_cfg["policy"]["network"][0]["activations"]

    gru_kwargs = dict(hidden_sizes=hidden_sizes, activation=activation,
                      temporal_dim=temporal_dim, gru_hidden=gru_hidden)
    models = {
        "policy": GRUActor(
            env.observation_space, env.action_space, env.device,
            clip_actions=model_cfg["policy"].get("clip_actions", False),
            clip_log_std=model_cfg["policy"].get("clip_log_std", True),
            min_log_std=model_cfg["policy"].get("min_log_std", -20.0),
            max_log_std=model_cfg["policy"].get("max_log_std", 2.0),
            **gru_kwargs,
        ),
        "value": GRUCritic(
            env.observation_space, env.action_space, env.device,
            clip_actions=model_cfg["value"].get("clip_actions", False),
            **gru_kwargs,
        ),
    }

    rollouts = agent_params["rollouts"]
    if memory_size is None:
        memory_size = rollouts if training else 1

    memory = RandomMemory(
        memory_size=memory_size, num_envs=env.num_envs, device=env.device,
        # skrl needs sequence_length for RNN training so it samples contiguous sequences
        **({"sequence_length": rollouts} if training else {}),
    )

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg.update({
        "state_preprocessor":        RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "value_preprocessor":        RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": env.device},
    })

    if training:
        cfg.update({
            "rollouts":                       rollouts,
            "learning_epochs":                agent_params["learning_epochs"],
            "mini_batches":                   agent_params["mini_batches"],
            "discount_factor":                agent_params["discount_factor"],
            "lambda":                         agent_params["lambda"],
            "learning_rate":                  agent_params["learning_rate"],
            "learning_rate_scheduler":        KLAdaptiveLR,
            "learning_rate_scheduler_kwargs": agent_params.get("learning_rate_scheduler_kwargs", {}),
            "random_timesteps":               agent_params.get("random_timesteps", 0),
            "learning_starts":                agent_params.get("learning_starts", 0),
            "grad_norm_clip":                 agent_params.get("grad_norm_clip", 1.0),
            "ratio_clip":                     agent_params.get("ratio_clip", 0.2),
            "value_clip":                     agent_params.get("value_clip", 0.2),
            "clip_predicted_values":          agent_params.get("clip_predicted_values", True),
            "entropy_loss_scale":             agent_params.get("entropy_loss_scale", 0.005),
            "value_loss_scale":               agent_params.get("value_loss_scale", 1.0),
            "kl_threshold":                   agent_params.get("kl_threshold", 0.0),
            "time_limit_bootstrap":           False,
            "experiment": {
                "directory":           agent_params["experiment"]["directory"],
                "experiment_name":     agent_params["experiment"]["experiment_name"],
                "write_interval":      1000,
                "checkpoint_interval": agent_params["experiment"].get("checkpoint_interval", 0),
            },
        })
        if agent_params.get("rewards_shaper_scale") is not None:
            scale = agent_params["rewards_shaper_scale"]
            cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * scale
        agent_cls = GRULoggedPPO
    else:
        cfg.update({"experiment": {"write_interval": 0, "checkpoint_interval": 0}})
        agent_cls = PPO

    return agent_cls(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


def build_sac_agent(env, agent_cfg, *, memory_size: int | None = None, training: bool = True):
    """Build a SAC agent.

    Args:
        memory_size: replay buffer size. Defaults to agent_cfg memory_size (training) or 1 (inference).
        training: if True, use full training config and LoggedSAC; if False, use minimal inference config and SAC.
    """
    agent_params = agent_cfg["agent"]
    model_cfg    = agent_cfg["models"]
    hidden_sizes = tuple(model_cfg["policy"]["network"][0]["layers"])
    activation   = model_cfg["policy"]["network"][0]["activations"]

    critic_kwargs = dict(
        hidden_sizes=hidden_sizes, activation=activation,
        clip_actions=model_cfg["critic"].get("clip_actions", False),
    )
    models = {
        "policy":        MLPActor(
            env.observation_space, env.action_space, env.device,
            hidden_sizes=hidden_sizes, activation=activation,
            clip_actions=model_cfg["policy"].get("clip_actions", False),
            clip_log_std=model_cfg["policy"].get("clip_log_std", True),
            min_log_std=model_cfg["policy"].get("min_log_std", -20.0),
            max_log_std=model_cfg["policy"].get("max_log_std", 2.0),
        ),
        "critic_1":        MLPSACCritic(env.observation_space, env.action_space, env.device, **critic_kwargs),
        "critic_2":        MLPSACCritic(env.observation_space, env.action_space, env.device, **critic_kwargs),
        "target_critic_1": MLPSACCritic(env.observation_space, env.action_space, env.device, **critic_kwargs),
        "target_critic_2": MLPSACCritic(env.observation_space, env.action_space, env.device, **critic_kwargs),
    }

    if memory_size is None:
        memory_size = agent_params["memory_size"] if training else 1

    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg.update({
        "state_preprocessor":        RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
    })

    if training:
        cfg.update({
            "gradient_steps":        agent_params.get("gradient_steps", 1),
            "batch_size":            agent_params["batch_size"],
            "discount_factor":       agent_params["discount_factor"],
            "polyak":                agent_params.get("polyak", 0.005),
            "actor_learning_rate":   agent_params["actor_learning_rate"],
            "critic_learning_rate":  agent_params["critic_learning_rate"],
            "entropy_learning_rate": agent_params["entropy_learning_rate"],
            "initial_entropy_value": agent_params.get("initial_entropy_value", 0.2),
            "target_entropy":        agent_params.get("target_entropy", None),
            "learn_entropy":         agent_params.get("learn_entropy", True),
            "random_timesteps":      agent_params.get("random_timesteps", 0),
            "learning_starts":       agent_params.get("learning_starts", 0),
            "grad_norm_clip":        agent_params.get("grad_norm_clip", 0),
            "experiment": {
                "directory":           agent_params["experiment"]["directory"],
                "experiment_name":     agent_params["experiment"]["experiment_name"],
                "write_interval":      1000,
                "checkpoint_interval": agent_params["experiment"].get("checkpoint_interval", 0),
            },
        })
        agent_cls = LoggedSAC
    else:
        cfg.update({"experiment": {"write_interval": 0, "checkpoint_interval": 0}})
        agent_cls = SAC

    return agent_cls(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
