# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali

from skrl.models.torch import DeterministicMixin, Model

from .mlp_actor_critic import MLPActor, _build_mlp


class MAPPOActor(MLPActor):
    """Parameter-shared actor for MAPPO; identical to MLPActor."""
    pass


class MAPPOCentralizedCritic(DeterministicMixin, Model):
    """Centralized critic for MAPPO.

    Receives the global state (all agents' observations concatenated) and outputs a
    scalar value estimate. The observation_space passed here must be the state space
    (typically 2x per-agent obs dim for two agents).
    """

    def __init__(self, observation_space, action_space, device,
                 hidden_sizes=(256, 256, 256), activation="elu", clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = _build_mlp(self.num_observations, 1, hidden_sizes, activation, layer_norm=True)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
