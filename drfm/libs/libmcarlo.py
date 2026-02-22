"""
File: libmcarlo
Use: Monte Carlo Policy
Update: Sun, 22 Feb 2026
Reference: github/dennybritz

Monte Carlo implementation with weighted importance sampling.
Monte Carlo: high variance, low bias. This means good convergence, even with
function approximation. Not sensitive to initial value.
"""

import numpy as np
from collections import defaultdict

class MDPMonteCarlo:
    """Monte Carlo Control for MDPs"""

    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(n_actions)) # state -> action val.
        self.C = defaultdict(lambda: np.zeros(n_actions)) # cum. weight denom.

    def random_policy(self) -> callable:
        """Uniform random policy"""
        probs = np.ones(self.n_actions) / self.n_actions
        def policy(state):
            return probs
        return policy

    def greedy_policy(self) -> callable:
        """Greedy policy"""
        def policy(state):
            probs = np.zeros(self.n_actions)
            best_action = np.argmax(self.Q[state])
            probs[best_action] = 1.0
            return probs

        return policy

    def control_off_policy(self, generate_episode: callable,
                          behavior_policy: callable,
                          num_episodes: int) -> tuple[dict, callable]:
        """Off-policy MC control using weighted importance sampling
        Finds optimal greedy policy using episodes from behavior policy.

        Args:
            generate_episode: Function that returns list of (s, a, r) tuples
            behavior_policy: Policy used to generate episodes
            num_episodes: Number of episodes to sample
        """
        target_policy = self.greedy_policy()

        for _ in range(num_episodes):
            episode = generate_episode()

            G = 0.0
            W = 1.0

            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward

                # Update weighted importance sampling
                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])

                # Stop if action differs from target policy
                if action != np.argmax(target_policy(state)):
                    break

                W *= 1.0 / behavior_policy(state)[action]

        return self.Q, target_policy

    def get_policy_state(self, state: int) -> np.ndarray:
        """Get action probabilities for a specific state"""
        probs = np.zeros(self.n_actions)
        best_action = np.argmax(self.Q[state])
        probs[best_action] = 1.0
        return probs

    def get_q_state(self, state: int) -> np.ndarray:
        """Get Q-values for a specific state"""
        return self.Q[state]
