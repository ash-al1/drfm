"""
File: Agent
Use: RL Agent Wrapper
Update: 
"""

import numpy as np
from libs.libmdp import MDP
from discrete import to_discrete


class Agent:
    def __init__(self, discreter: to_discrete, n_actions: int, gamma: float,
                 epsilon: float = 0.1):
        """
        Args:
            to_discrete: Map continuous outputs (isaac) -> discrete (mdp)
            n_actions: Total number of actions
            gamma: [0,1]
            epsilon: Exploration rate
        """
        self.disc = discreter
        self.n_states = discreter.n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon

        # counts for estimating P(s'|s,a) and R(s,a)
        self.transition_counts = np.zeros((self.n_states, n_actions, self.n_states))
        self.reward_sum = np.zeros((self.n_states, n_actions))
        self.reward_count = np.zeros((self.n_states, n_actions))

        # π: uniform random
        self.policy = np.ones((self.n_states, n_actions)) / n_actions
        self.V = np.zeros(self.n_states)
        self.Q = np.zeros((self.n_states, n_actions))

    def get_state(self, obs: np.ndarray) -> int:
        return self.disc.discretize(obs)

    def get_states(self, obs_batch: np.ndarray) -> np.ndarray:
        return self.disc.discretize_batch(obs_batch)

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.policy[state]))

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        """Vectorized epsilon-greedy action selection for multiple states"""
        num_envs = len(states)

        greedy_actions = np.argmax(self.policy[states], axis=1)
        explore_mask = np.random.random(num_envs) < self.epsilon
        random_actions = np.random.randint(0, self.n_actions, size=num_envs)
        actions = np.where(explore_mask, random_actions, greedy_actions)

        return actions

    def update_model(self, state: int, action: int, reward: float, next_state: int):
        self.transition_counts[state, action, next_state] += 1
        self.reward_sum[state, action] += reward
        self.reward_count[state, action] += 1

    def update_model_batch(self, states: np.ndarray, actions: np.ndarray,
                           rewards: np.ndarray, next_states: np.ndarray):
        """Vectorized update"""
        np.add.at(self.transition_counts, (states, actions, next_states), 1)
        np.add.at(self.reward_sum, (states, actions), rewards)
        np.add.at(self.reward_count, (states, actions), 1)

    def build_mdp(self) -> MDP:
        """(Vectorized) Construct an MDP from accumulated experience"""
        totals = np.sum(self.transition_counts, axis=2)
        P = np.full((self.n_states, self.n_actions, self.n_states), 1.0 / self.n_states)
        np.divide(
            self.transition_counts,
            totals[:, :, np.newaxis],
            out=P,
            where=totals[:, :, np.newaxis] > 0
        )

        R = np.zeros((self.n_states, self.n_actions))
        np.divide(
            self.reward_sum,
            self.reward_count,
            out=R,
            where=self.reward_count > 0
        )

        return MDP(P, R, self.gamma)

    # Swap policy/value iteration
    # TODO: should create parameter arg to change policy/value iter later
    def solve(self):
        mdp = self.build_mdp()
        #self.V, self.Q, self.policy = mdp.policy_iter()
        self.V, self.Q, self.policy = mdp.value_iter()

    def save(self, policy_path: str, values_path: str):
        np.save(policy_path, self.policy)
        np.save(values_path, self.V)

    # TODO: Use this load function for resuming training
    def load(self, policy_path: str, values_path: str):
        self.policy = np.load(policy_path)
        self.V = np.load(values_path)
