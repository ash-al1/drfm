import numpy as np


class to_discrete:
    def __init__(self, obs_bins: list[int], obs_ranges: list[tuple[float, float]]):
        self.obs_bins = obs_bins
        self.obs_ranges = obs_ranges
        self.n_dims = len(obs_bins)
        self.n_states = int(np.prod(obs_bins))
        self.edges = [np.linspace(lo, hi, b + 1)[1:-1] for (lo, hi), b in zip(obs_ranges, obs_bins)]

    def discretize(self, obs: np.ndarray) -> int:
        indices = [
            int(np.digitize(np.clip(obs[i], *self.obs_ranges[i]), self.edges[i]))
            for i in range(self.n_dims)
        ]
        state = 0
        for i in range(self.n_dims):
            state = state * self.obs_bins[i] + indices[i]
        return state

    def discretize_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        clipped = np.stack([
            np.clip(obs_batch[:, i], *self.obs_ranges[i]) for i in range(self.n_dims)
        ], axis=1)
        indices = np.stack([
            np.digitize(clipped[:, i], self.edges[i]) for i in range(self.n_dims)
        ], axis=1)
        states = np.zeros(obs_batch.shape[0], dtype=np.int32)
        for i in range(self.n_dims):
            states = states * self.obs_bins[i] + indices[:, i]
        return states
