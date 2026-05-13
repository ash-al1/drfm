import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

_ACTIVATIONS = {
    "elu":        nn.ELU,
    "relu":       nn.ReLU,
    "tanh":       nn.Tanh,
    "sigmoid":    nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "selu":       nn.SELU,
}


def _build_mlp(input_size: int, output_size: int, hidden_sizes: tuple, activation: str, layer_norm: bool = False) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation.lower()]
    layers = []
    prev = input_size
    for i, h in enumerate(hidden_sizes):
        if layer_norm and i > 0:
            layers.append(nn.LayerNorm(prev))
        layers.extend([nn.Linear(prev, h), act_cls()])
        prev = h
    if layer_norm:
        layers.append(nn.LayerNorm(prev))
    layers.append(nn.Linear(prev, output_size))
    return nn.Sequential(*layers)


class SharedMLP(GaussianMixin, DeterministicMixin, Model):

    def __init__(self, observation_space, action_space, device,
                 hidden_sizes=(256, 256, 256), activation="elu"):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                                min_log_std=-20.0, max_log_std=2.0, reduction="sum", role="policy")
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        obs_dim = self.num_observations
        act_dim = self.num_actions
        feat_dim = hidden_sizes[-1]

        act_cls = _ACTIVATIONS[activation.lower()]
        layers = []
        prev = obs_dim
        for i, h in enumerate(hidden_sizes):
            if i > 0:
                layers.append(nn.LayerNorm(prev))
            layers.extend([nn.Linear(prev, h), act_cls()])
            prev = h
        layers.append(nn.LayerNorm(prev))
        self.backbone = nn.Sequential(*layers)

        self.policy_head = nn.Linear(feat_dim, act_dim)
        self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(feat_dim, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        features = self.backbone(inputs["observations"])
        if role == "policy":
            return self.policy_head(features), {"log_std": self.log_std_parameter}
        return self.value_head(features), {}


class MLPActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 hidden_sizes=(256, 256, 256), activation="elu",
                 clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std, min_log_std=min_log_std, max_log_std=max_log_std)
        self.net = _build_mlp(self.num_observations, self.num_actions, hidden_sizes, activation, layer_norm=True)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class MLPCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 hidden_sizes=(256, 256, 256), activation="elu", clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = _build_mlp(self.num_observations, 1, hidden_sizes, activation, layer_norm=True)

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class MLPSACCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 hidden_sizes=(256, 256, 256), activation="elu", clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        input_dim = self.num_observations + self.num_actions
        self.net = _build_mlp(input_dim, 1, hidden_sizes, activation, layer_norm=True)

    def compute(self, inputs, role):
        x = torch.cat([inputs["observations"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}
