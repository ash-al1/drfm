# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from models.architectures.mlp_actor_critic import _build_mlp


class _GRUBase(Model):
    """Shared GRU encoder that splits obs into temporal (GRU) and static (passthrough) streams."""

    _GRU_HIDDEN = 64

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes: tuple = (256, 256),
        activation: str = "elu",
        temporal_dim: int = 32,
        gru_hidden: int = 64,
        num_envs: int = 1,
    ):
        super().__init__(observation_space=observation_space, action_space=action_space, device=device)

        self._temporal_dim = temporal_dim
        self._gru_hidden = gru_hidden
        self._num_envs = num_envs

        obs_dim = self.num_observations
        static_dim = obs_dim - temporal_dim

        self.gru = nn.GRU(input_size=temporal_dim, hidden_size=gru_hidden, num_layers=1, batch_first=True)
        self._mlp_input_dim = gru_hidden + static_dim

    def get_specification(self) -> dict:
        """Return RNN spec so PPO_RNN can manage hidden states."""
        return {"rnn": {"sizes": [(1, self._num_envs, self._gru_hidden)], "sequence_length": 1}}

    def _gru_forward(self, inputs: dict) -> tuple[torch.Tensor, list]:
        """Run GRU on temporal stream; return (combined_features, [new_hidden])."""
        states = inputs["observations"]
        temporal = states[..., : self._temporal_dim]
        static   = states[..., self._temporal_dim :]

        rnn_states = inputs.get("rnn", [])
        if rnn_states:
            hidden = rnn_states[0]
        else:
            hidden = torch.zeros(1, states.shape[0], self._gru_hidden, device=self.device)

        if temporal.dim() == 2:
            temporal = temporal.unsqueeze(1)  # (N, 1, temporal_dim)

        gru_out, new_hidden = self.gru(temporal, hidden)

        if states.dim() == 2:
            gru_out = gru_out[:, -1, :]  # (N, gru_hidden) - take last timestep

        gru_feat = gru_out if gru_out.dim() == 2 else gru_out.reshape(-1, self._gru_hidden)

        if static.dim() == 3:
            static = static.reshape(-1, static.shape[-1])

        combined = torch.cat([gru_feat, static], dim=-1)
        return combined, [new_hidden]


class GRUActor(GaussianMixin, _GRUBase):
    """Gaussian actor with GRU temporal encoder."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes: tuple = (256, 256),
        activation: str = "elu",
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        temporal_dim: int = 32,
        gru_hidden: int = 64,
        num_envs: int = 1,
    ):
        _GRUBase.__init__(self, observation_space, action_space, device,
                          hidden_sizes=hidden_sizes, activation=activation,
                          temporal_dim=temporal_dim, gru_hidden=gru_hidden, num_envs=num_envs)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)

        act_dim = self.num_actions
        self.net = _build_mlp(self._mlp_input_dim, act_dim, hidden_sizes, activation, layer_norm=True)
        # Start policy output near zero = hover
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)
        self.log_std_parameter = nn.Parameter(torch.full((act_dim,), -1.0))  # std≈0.37, gentler exploration

    def compute(self, inputs, role):
        features, rnn_out = self._gru_forward(inputs)
        return self.net(features), {"log_std": self.log_std_parameter, "rnn": rnn_out}


class GRUCritic(DeterministicMixin, _GRUBase):
    """Deterministic critic with GRU temporal encoder."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes: tuple = (256, 256),
        activation: str = "elu",
        clip_actions: bool = False,
        temporal_dim: int = 32,
        gru_hidden: int = 64,
        num_envs: int = 1,
    ):
        _GRUBase.__init__(self, observation_space, action_space, device,
                          hidden_sizes=hidden_sizes, activation=activation,
                          temporal_dim=temporal_dim, gru_hidden=gru_hidden, num_envs=num_envs)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = _build_mlp(self._mlp_input_dim, 1, hidden_sizes, activation, layer_norm=True)

    def compute(self, inputs, role):
        features, rnn_out = self._gru_forward(inputs)
        return self.net(features), {"rnn": rnn_out}
