# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali

import copy

import torch

from skrl.agents.torch.base import ExperimentCfg
from skrl.agents.torch.ppo import PPO, PPO_CFG, PPO_RNN
from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from models.architectures.mlp_actor_critic import MLPActor, MLPCritic, MLPSACCritic
from models.architectures.mappo_actor_critic import MAPPOActor, MAPPOCentralizedCritic
from models.architectures.rnn_actor_critic import GRUActor, GRUCritic


class GRULoggedPPO(PPO_RNN):
    """PPO_RNN subclass that logs metrics; hidden state resets are handled by PPO_RNN internally."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_metrics = {}
        self._update_count = 0

    def record_transition(self, *, observations, states, actions, rewards, next_observations, next_states,
                          terminated, truncated, infos, timestep, timesteps):
        if timestep == 0:
            print(f"[GRU-DEBUG] obs shape: {observations.shape}, actions shape: {actions.shape}")
            spec = self.policy.get_specification()
            print(f"[GRU-DEBUG] policy RNN spec: {spec}")
            print(f"[GRU-DEBUG] self._rnn: {self._rnn}, seq_len: {self._rnn_sequence_length}")
            print(f"[GRU-DEBUG] rnn_initial_states keys: {list(self._rnn_initial_states.keys())}, "
                  f"policy states: {[s.shape for s in self._rnn_initial_states['policy']]}")
        super().record_transition(
            observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states,
            terminated=terminated, truncated=truncated, infos=infos,
            timestep=timestep, timesteps=timesteps,
        )

    def _update(self, timestep, timesteps):
        self._update_count += 1
        PPO_RNN._update(self, timestep, timesteps)
        self.last_metrics = {k: list(v) for k, v in self.tracking_data.items()}


class LoggedPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_metrics = {}
        self._update_count = 0

    def record_transition(self, *, observations, states, actions, rewards, next_observations, next_states,
                          terminated, truncated, infos, timestep, timesteps):
        if timestep == 0:
            print(f"[PPO-DEBUG] obs shape: {observations.shape}, actions shape: {actions.shape}")
        super().record_transition(
            observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states,
            terminated=terminated, truncated=truncated, infos=infos,
            timestep=timestep, timesteps=timesteps,
        )

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

    cfg = PPO_CFG(
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": env.observation_space, "device": env.device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": env.device},
    )

    if training:
        cfg.rollouts = agent_params["rollouts"]
        cfg.learning_epochs = agent_params["learning_epochs"]
        cfg.mini_batches = agent_params["mini_batches"]
        cfg.discount_factor = agent_params["discount_factor"]
        cfg.gae_lambda = agent_params["lambda"]
        cfg.learning_rate = agent_params["learning_rate"]
        cfg.learning_rate_scheduler = KLAdaptiveLR
        cfg.learning_rate_scheduler_kwargs = agent_params.get("learning_rate_scheduler_kwargs", {})
        cfg.random_timesteps = agent_params.get("random_timesteps", 0)
        cfg.learning_starts = agent_params.get("learning_starts", 0)
        cfg.grad_norm_clip = agent_params.get("grad_norm_clip", 1.0)
        cfg.ratio_clip = agent_params.get("ratio_clip", 0.2)
        cfg.value_clip = agent_params.get("value_clip", 0.2)
        cfg.entropy_loss_scale = agent_params.get("entropy_loss_scale", 0.005)
        cfg.value_loss_scale = agent_params.get("value_loss_scale", 1.0)
        cfg.kl_threshold = agent_params.get("kl_threshold", 0.0)
        cfg.time_limit_bootstrap = False
        cfg.experiment = ExperimentCfg(
            directory=agent_params["experiment"]["directory"],
            experiment_name=agent_params["experiment"]["experiment_name"],
            write_interval=1000,
            checkpoint_interval=agent_params["experiment"].get("checkpoint_interval", 0),
        )
        if agent_params.get("rewards_shaper_scale") is not None:
            scale = agent_params["rewards_shaper_scale"]
            cfg.rewards_shaper = lambda rewards, *args, **kwargs: rewards * scale
        agent_cls = LoggedPPO
    else:
        cfg.experiment = ExperimentCfg(write_interval=0, checkpoint_interval=0)
        agent_cls = PPO

    return agent_cls(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


def build_ppo_gru_agent(env, agent_cfg, *, memory_size: int | None = None, training: bool = True,
                        temporal_dim: int = 32, gru_hidden: int = 64):
    """Build a PPO_RNN agent with GRU actor and critic.

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
            num_envs=env.num_envs,
            **gru_kwargs,
        ),
        "value": GRUCritic(
            env.observation_space, env.action_space, env.device,
            clip_actions=model_cfg["value"].get("clip_actions", False),
            num_envs=env.num_envs,
            **gru_kwargs,
        ),
    }

    rollouts = agent_params["rollouts"]
    if memory_size is None:
        memory_size = rollouts if training else 1

    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    cfg = PPO_CFG(
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": env.observation_space, "device": env.device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": env.device},
    )

    if training:
        cfg.rollouts = rollouts
        cfg.learning_epochs = agent_params["learning_epochs"]
        cfg.mini_batches = agent_params["mini_batches"]
        cfg.discount_factor = agent_params["discount_factor"]
        cfg.gae_lambda = agent_params["lambda"]
        cfg.learning_rate = agent_params["learning_rate"]
        cfg.learning_rate_scheduler = KLAdaptiveLR
        cfg.learning_rate_scheduler_kwargs = agent_params.get("learning_rate_scheduler_kwargs", {})
        cfg.random_timesteps = agent_params.get("random_timesteps", 0)
        cfg.learning_starts = agent_params.get("learning_starts", 0)
        cfg.grad_norm_clip = agent_params.get("grad_norm_clip", 1.0)
        cfg.ratio_clip = agent_params.get("ratio_clip", 0.2)
        cfg.value_clip = agent_params.get("value_clip", 0.2)
        cfg.entropy_loss_scale = agent_params.get("entropy_loss_scale", 0.005)
        cfg.value_loss_scale = agent_params.get("value_loss_scale", 1.0)
        cfg.kl_threshold = agent_params.get("kl_threshold", 0.0)
        cfg.time_limit_bootstrap = False
        cfg.experiment = ExperimentCfg(
            directory=agent_params["experiment"]["directory"],
            experiment_name=agent_params["experiment"]["experiment_name"],
            write_interval=1000,
            checkpoint_interval=agent_params["experiment"].get("checkpoint_interval", 0),
        )
        if agent_params.get("rewards_shaper_scale") is not None:
            scale = agent_params["rewards_shaper_scale"]
            cfg.rewards_shaper = lambda rewards, *args, **kwargs: rewards * scale
        agent_cls = GRULoggedPPO
    else:
        cfg.experiment = ExperimentCfg(write_interval=0, checkpoint_interval=0)
        agent_cls = PPO_RNN

    return agent_cls(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


class LoggedMAPPO(MAPPO):
    """MAPPO subclass that captures training metrics after each update."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_metrics = {}
        self._update_count = 0

    def _update(self, timestep, timesteps):
        self._update_count += 1
        MAPPO._update(self, timestep, timesteps)
        self.last_metrics = {k: list(v) for k, v in self.tracking_data.items()}


def build_mappo_agent(env, agent_cfg, *, memory_size: int | None = None, training: bool = True):
    """Build a MAPPO agent with parameter-shared actor and centralized critic.

    Args:
        env: Skrl-wrapped MARL environment exposing possible_agents, observation_spaces,
             action_spaces, and state_spaces.
        agent_cfg: Agent configuration dict (from skrl_mappo_cfg.yaml).
        memory_size: Rollout buffer size. Defaults to agent_cfg rollouts (training) or 1.
        training: If True, configure for training with LoggedMAPPO; else use MAPPO.
    """
    agent_params = agent_cfg["agent"]
    model_cfg = agent_cfg["models"]
    hidden_sizes = tuple(model_cfg["policy"]["network"][0]["layers"])
    activation = model_cfg["policy"]["network"][0]["activations"]

    possible_agents = env.possible_agents
    obs_space_0 = env.observation_spaces[possible_agents[0]]
    act_space_0 = env.action_spaces[possible_agents[0]]
    state_space_0 = env.state_spaces[possible_agents[0]]

    # One shared actor instance (parameter sharing across agents)
    shared_actor = MAPPOActor(
        obs_space_0, act_space_0, env.device,
        hidden_sizes=hidden_sizes, activation=activation,
        clip_actions=model_cfg["policy"].get("clip_actions", False),
        clip_log_std=model_cfg["policy"].get("clip_log_std", True),
        min_log_std=model_cfg["policy"].get("min_log_std", -20.0),
        max_log_std=model_cfg["policy"].get("max_log_std", 2.0),
    )
    # Centralized critic takes concatenated all-agent observations as state
    centralized_critic = MAPPOCentralizedCritic(
        state_space_0, act_space_0, env.device,
        hidden_sizes=hidden_sizes, activation=activation,
        clip_actions=model_cfg["value"].get("clip_actions", False),
    )
    models = {a: {"policy": shared_actor, "value": centralized_critic} for a in possible_agents}

    if memory_size is None:
        memory_size = agent_params["rollouts"] if training else 1
    memories = {
        a: RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)
        for a in possible_agents
    }

    cfg = MAPPO_CFG(
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": obs_space_0, "device": env.device},
        state_preprocessor=RunningStandardScaler,
        state_preprocessor_kwargs={"size": state_space_0, "device": env.device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": env.device},
        learning_rate_scheduler_kwargs=None,
    )

    if training:
        cfg.rollouts = agent_params["rollouts"]
        cfg.learning_epochs = agent_params["learning_epochs"]
        cfg.mini_batches = agent_params["mini_batches"]
        cfg.discount_factor = agent_params["discount_factor"]
        cfg.gae_lambda = agent_params.get("gae_lambda", agent_params.get("lambda", 0.95))
        cfg.learning_rate = agent_params["learning_rate"]
        cfg.learning_rate_scheduler = KLAdaptiveLR
        cfg.learning_rate_scheduler_kwargs = agent_params.get("learning_rate_scheduler_kwargs") or None
        cfg.random_timesteps = agent_params.get("random_timesteps", 0)
        cfg.learning_starts = agent_params.get("learning_starts", 0)
        cfg.grad_norm_clip = agent_params.get("grad_norm_clip", 0.5)
        cfg.ratio_clip = agent_params.get("ratio_clip", 0.2)
        cfg.value_clip = agent_params.get("value_clip", 0.2)
        cfg.entropy_loss_scale = agent_params.get("entropy_loss_scale", 0.0)
        cfg.value_loss_scale = agent_params.get("value_loss_scale", 2.5)
        cfg.kl_threshold = agent_params.get("kl_threshold", 0.0)
        cfg.time_limit_bootstrap = agent_params.get("time_limit_bootstrap", False)
        cfg.experiment = ExperimentCfg(
            directory=agent_params["experiment"]["directory"],
            experiment_name=agent_params["experiment"]["experiment_name"],
            write_interval=agent_params["experiment"].get("write_interval", 1000),
            checkpoint_interval=agent_params["experiment"].get("checkpoint_interval", 0),
        )
        if agent_params.get("rewards_shaper_scale") is not None:
            scale = agent_params["rewards_shaper_scale"]
            cfg.rewards_shaper = lambda rewards, *args, **kwargs: rewards * scale
        agent_cls = LoggedMAPPO
    else:
        cfg.experiment = ExperimentCfg(write_interval=0, checkpoint_interval=0)
        agent_cls = MAPPO

    return agent_cls(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        cfg=cfg,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        state_spaces=env.state_spaces,
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

    cfg = SAC_CFG(
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": env.observation_space, "device": env.device},
    )

    if training:
        cfg.learning_rate = [
            agent_params["actor_learning_rate"],
            agent_params["critic_learning_rate"],
            agent_params["entropy_learning_rate"],
        ]
        cfg.gradient_steps = agent_params.get("gradient_steps", 1)
        cfg.batch_size = agent_params["batch_size"]
        cfg.discount_factor = agent_params["discount_factor"]
        cfg.polyak = agent_params.get("polyak", 0.005)
        cfg.learn_entropy = agent_params.get("learn_entropy", True)
        cfg.initial_entropy_value = agent_params.get("initial_entropy_value", 0.2)
        cfg.target_entropy = agent_params.get("target_entropy", None)
        cfg.random_timesteps = agent_params.get("random_timesteps", 0)
        cfg.learning_starts = agent_params.get("learning_starts", 0)
        cfg.grad_norm_clip = agent_params.get("grad_norm_clip", 0)
        cfg.experiment = ExperimentCfg(
            directory=agent_params["experiment"]["directory"],
            experiment_name=agent_params["experiment"]["experiment_name"],
            write_interval=1000,
            checkpoint_interval=agent_params["experiment"].get("checkpoint_interval", 0),
        )
        agent_cls = LoggedSAC
    else:
        cfg.experiment = ExperimentCfg(write_interval=0, checkpoint_interval=0)
        agent_cls = SAC

    return agent_cls(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
