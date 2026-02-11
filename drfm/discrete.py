"""
File: discrete
Use: 
Update: 
"""

import numpy as np


class to_discrete:
    def __init__(self, obs_bins: list[int], obs_ranges: list[tuple[float, float]]):
        """
        Args:
            obs_bins: number of bins per observation dimension, e.g. [10, 10, 10, 10]
            obs_ranges: (low, high) clipping range per dimension, e.g.
                        [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.0, 3.0)]
        """
        self.obs_bins = obs_bins
        self.obs_ranges = obs_ranges
        self.n_dims = len(obs_bins)
        self.n_states = int(np.prod(obs_bins))

        # precompute bin edges per dimension
        self.edges = []
        for i in range(self.n_dims):
            low, high = obs_ranges[i]
            self.edges.append(np.linspace(low, high, obs_bins[i] + 1)[1:-1])

    def discretize(self, obs: np.ndarray) -> int:
        """Convert a continuous observation vector into a single state index."""
        indices = []
        for i in range(self.n_dims):
            low, high = self.obs_ranges[i]
            val = np.clip(obs[i], low, high)
            idx = int(np.digitize(val, self.edges[i]))
            indices.append(idx)

        # flatten multi-dim index to single int
        state = 0
        for i in range(self.n_dims):
            state = state * self.obs_bins[i] + indices[i]
        return state

    def discretize_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Vectorized discretization for (num_envs, n_dims) observations.

        Args:
            obs_batch: shape (num_envs, n_dims)

        Returns:
            state_indices: shape (num_envs,) of discrete state indices
        """
        num_envs = obs_batch.shape[0]

        # clip all observations at once
        clipped = np.empty_like(obs_batch)
        for i in range(self.n_dims):
            low, high = self.obs_ranges[i]
            clipped[:, i] = np.clip(obs_batch[:, i], low, high)

        # bulk digitize for each dimension
        indices = np.empty((num_envs, self.n_dims), dtype=np.int32)
        for i in range(self.n_dims):
            indices[:, i] = np.digitize(clipped[:, i], self.edges[i])

        # vectorized flattening: state = indices[0]*b1*b2*b3 + indices[1]*b2*b3 + ...
        states = np.zeros(num_envs, dtype=np.int32)
        for i in range(self.n_dims):
            states = states * self.obs_bins[i] + indices[:, i]

        return states
