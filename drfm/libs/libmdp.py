"""
File: libmdp
Use: Markov Decision Process
Update: Wed, 11 Feb 2026 11:23:40
"""

import numpy as np

class MDP:
    def __init__(self, transition: np.ndarray, reward: np.ndarray,
                 gamma: float, mu: np.ndarray = None):
        """MDP with transition matrix and rewards

        Args:
            transitions: P(s'|s,a)
            reward: R(s,a)
            gamma: [0, 1]
            mu: uniform state distribution
        """
        self.P = transition
        self.R = reward
        self.gamma = gamma

        # P(s'|s,a)
        self.n_states, self.n_actions, _ = transition.shape
        self.mu = mu if mu is not None else np.ones(self.n_states) / self.n_states
        
    def q(self, V: np.ndarray) -> np.array:
        """Action value function 

        Bellmans equation: Q(s,a) = R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')

        The (MAC) expectation from taking an action a in state s w.r.t greedy π

        Refs:
        https://en.wikipedia.org/wiki/Bellman_equation#The_Bellman_equation
        """
        return self.R + self.gamma * np.einsum('ijk,k->ij', self.P, V)

    def v(self, Q: np.ndarray, policy: np.ndarray) -> np.ndarray:
        """State value function

        V^π(s) = Σ_a π(a|s) * Q^π(s,a)

        Expected return from state s following policy π
        """
        return np.sum(policy * Q, axis=1)
    
    def policy(self, Q: np.ndarray) -> np.ndarray:
        """Map states to action probabilities

        π*(s) = argmax_a Q(s,a)
        """
        policy = np.zeros((self.n_states, self.n_actions))
        best_actions = np.argmax(Q, axis=1)
        policy[np.arange(self.n_states), best_actions] = 1.0
        return policy

    def policy_eval(self, policy: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Using iterative application (not fastest this is work in progress)
        
        Iterates: V_{k+1}(s) = Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V_k(s')]
        """
        V = np.zeros(self.n_states)
        while True:
            Q = self.q(V)
            Vz = self.v(Q, policy)
            # Check convergence
            if np.max(np.abs(Vz - V)) < tol:
                return Vz
            V = Vz
    
    def value_iter(self, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find optimal V, Q, and π using Bellman optimality equation

        V_{k+1}(s) = max_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V_k(s')], where
        Bellmans optimality: V*(s) = max_a Q*(s,a)

        Refs:
        https://en.wikipedia.org/wiki/Bellman_equation#Bellman's_principle_of_optimality
        https://en.wikipedia.org/wiki/Dynamic_programming#Dijkstra's_algorithm_for_the_shortest_path_problem
        """
        V = np.zeros(self.n_states)
        while True:
            Q = self.q(V)
            Vz = np.max(Q, axis=1)
            # Convergence; where π* is greedy policy
            if np.max(np.abs(Vz - V)) < tol:
                return Vz, Q, self.policy(Q)
            V = Vz

    def policy_iter(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute V^π then update π(s) using argmax_a Q^π(s,a)

        Initialize using uniform random policy since policy doesn't matter,
        algorithm will learn it anyways
        """
        policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        while True:
            V = self.policy_eval(policy)
            Q = self.q(V)
            policyz = self.policy(Q)
            # If policy is unchanged then reached optimal
            if np.array_equal(policy, policyz):
                return V, Q, policyz
            policy = policyz
