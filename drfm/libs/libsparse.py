"""
File: libsparse
Use: Sparse representation for better memory handling
Update: Wed, 11 Feb 2026 08:13:10
"""

import numpy as np

class SparseMDP:
    """Sparse MDP

    Transitions: Stores transitions as dictionaries {s: {a: (s', r)}}
    """
    def __init__(self, transitions, n_states, n_actions, gamma):
        """Init

        Args:
            transitions: Dict{}
            n_states: Total number of states
            n_actions: Total number of actions
            gamma: [0,1]
        """
        self.transitions = transitions
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

    def value_iter(self, tol=1e-6, max_iters=10000):
        """Value iteration for sparse & deterministic transitions
        
        Bellman Optimality: V*(s) = max_a [R(s,a) + γ * V*(s')]
        """
        V = np.zeros(self.n_states)

        for iteration in range(max_iters):
            V_new = np.zeros(self.n_states)

            for s in range(self.n_states):
                # If unreachable
                if s not in self.transitions:
                    continue

                best_val = float('-inf')
                # Max overall actions from state s
                for a in range(self.n_actions):
                    next_s, reward = self.transitions[s][a]
                    val = reward + self.gamma * V[next_s]
                    if val > best_val:
                        best_val = val

                V_new[s] = best_val

            # Convergence
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new

        # Greedy policy from converged value function
        policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            if s not in self.transitions:
                continue
            
            # Best action that achieves V*(S)
            best_a = 0
            best_val = float('-inf')
            for a in range(self.n_actions):
                next_s, reward = self.transitions[s][a]
                val = reward + self.gamma * V[next_s]
                if val > best_val:
                    best_val = val
                    best_a = a
            policy[s, best_a] = 1.0

        return V, policy

    def policy_eval(self, policy, tol=1e-6, max_iters=10000):
        """Policy evaluation"""
        V = np.zeros(self.n_states)

        for iteration in range(max_iters):
            V_new = np.zeros(self.n_states)

            for s in range(self.n_states):
                if s not in self.transitions:
                    continue

                for a in range(self.n_actions):
                    if policy[s, a] > 0:
                        next_s, reward = self.transitions[s][a]
                        V_new[s] = reward + self.gamma * V[next_s]
                        break

            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new

        return V

    def extract_policy(self, V):
        """Extract greedy policy from value function"""
        policy = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            if s not in self.transitions:
                continue

            best_a = 0
            best_val = float('-inf')
            for a in range(self.n_actions):
                next_s, reward = self.transitions[s][a]
                val = reward + self.gamma * V[next_s]
                if val > best_val:
                    best_val = val
                    best_a = a
            policy[s, best_a] = 1.0

        return policy

    def policy_iter(self, tol=1e-6, max_iters=10000):
        """Policy iteration"""
        policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            if s in self.transitions:
                policy[s, 0] = 1.0

        for iteration in range(max_iters):
            V = self.policy_eval(policy)
            policy_new = self.extract_policy(V)

            if np.array_equal(policy, policy_new):
                break
            policy = policy_new

        return V, policy
