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
    ):
        super().__init__(observation_space, action_space, device)

        self._temporal_dim = temporal_dim
        self._gru_hidden = gru_hidden

        obs_dim = self.observation_space.shape[0]
        static_dim = obs_dim - temporal_dim

        self.gru = nn.GRU(input_size=temporal_dim, hidden_size=gru_hidden, num_layers=1, batch_first=True)
        self._mlp_input_dim = gru_hidden + static_dim

        # hidden state: [num_layers=1, num_envs, gru_hidden], initialized on first forward pass
        self._hidden: torch.Tensor | None = None

    def reset_hidden(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero GRU hidden state for the given env indices, or all if env_ids is None."""
        if self._hidden is None:
            return
        if env_ids is None:
            self._hidden.zero_()
        else:
            self._hidden[:, env_ids, :] = 0.0

    def _init_hidden(self, num_envs: int) -> None:
        self._hidden = torch.zeros(1, num_envs, self._gru_hidden, device=self.device)

    def _gru_forward(self, states: torch.Tensor) -> torch.Tensor:
        """Run GRU on temporal stream; return concatenated [gru_out, static]."""
        temporal = states[:, : self._temporal_dim]   # (N, temporal_dim)
        static   = states[:, self._temporal_dim :]   # (N, static_dim)

        num_envs = states.shape[0]
        if self._hidden is None or self._hidden.shape[1] != num_envs:
            self._init_hidden(num_envs)

        # GRU expects (batch, seq, features); we have seq=1 at inference time
        gru_out, self._hidden = self.gru(
            temporal.unsqueeze(1),       # (N, 1, temporal_dim)
            self._hidden,                # (1, N, gru_hidden)
        )
        # detach hidden from graph between steps to avoid BPTT across episodes
        self._hidden = self._hidden.detach()

        gru_feat = gru_out.squeeze(1)    # (N, gru_hidden)
        return torch.cat([gru_feat, static], dim=-1)  # (N, gru_hidden + static_dim)


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
    ):
        _GRUBase.__init__(self, observation_space, action_space, device,
                          hidden_sizes=hidden_sizes, activation=activation,
                          temporal_dim=temporal_dim, gru_hidden=gru_hidden)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        act_dim = self.action_space.shape[0]
        self.net = _build_mlp(self._mlp_input_dim, act_dim, hidden_sizes, activation, layer_norm=True)
        self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

    def compute(self, inputs, role):
        features = self._gru_forward(inputs["states"])
        return self.net(features), self.log_std_parameter, {}


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
    ):
        _GRUBase.__init__(self, observation_space, action_space, device,
                          hidden_sizes=hidden_sizes, activation=activation,
                          temporal_dim=temporal_dim, gru_hidden=gru_hidden)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = _build_mlp(self._mlp_input_dim, 1, hidden_sizes, activation, layer_norm=True)

    def compute(self, inputs, role):
        features = self._gru_forward(inputs["states"])
        return self.net(features), {}
