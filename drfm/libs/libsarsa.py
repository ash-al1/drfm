"""
File: libsarsa
Use: SARSA
Update: Sat, 21 Feb 2026
Reference: github/dennybritz

SARSA(0):  1 step on policy
SARSA(n):  n steps, on policy
SARSA(λ):  Eligibility traces. Backward view. On policy.
"""

import numpy as np
from collections import defaultdict

class SARSA:
    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.99,
                 alpha: float = 0.5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def epsilon_greedy_policy(self, epsilon: float) -> callable:
        def policy(state):
            probs = np.ones(self.n_actions, dtype=float) * epsilon / self.n_actions
            best_action = np.argmax(self.Q[state])
            probs[best_action] += 1.0 - epsilon
            return probs
        return policy

    def sarsa0(self, generate_episode: callable, num_episodes: int,
               epsilon: float = 0.1) -> tuple[dict, callable]:
        """SARSA(0)

        Args:
            generate_episode: Callable that returns (s, a, r, s', done) tuple
            num_episodes: self-evident eh?
            epsilon: Exploration alpha

        Returns:
            Q: Action value function
            π: epsilon-greedy policy
        """
        policy = self.epsilon_greedy_policy(epsilon)
        for i in range(num_episodes):
            if (i+1) % 1000 == 0:
                print(f"\rEpisode {i}/{num_episodes}")

            episode = generate_episode()

            for t in range((len(episode) - 1)):
                state, action, reward, next_state, done = episode[t]
                next_action_probs = policy(next_state)
                next_action = np.random.choice(self.n_actions,
                                               p=next_action_probs)

                # (s', a')
                if done:
                    td_target = reward
                else:
                    td_target = reward + (self.gamma *
                                          self.Q[next_state][next_action])
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta

        return self.Q, policy

    def sarsan(self, generate_episode: callable, num_episodes: int,
               epsilon: float = 0.1) -> tuple[dict, callable]:
        """SARSA(N)

        Args:
            generate_episode: Callable that returns (s, a, r, s', done) tuple
            num_episodes: self-evident eh?
            epsilon: Exploration alpha

        Returns:
            Q: Action value function
            π: epsilon-greedy policy
        """
        policy = self.epsilon_greedy_policy(epsilon)
        for i in range(num_episodes):
            if (i+1) % 1000 == 0:
                print(f"\rEpisode {i}/{num_episodes}")

            episode = generate_episode()
            T = len(episode)

            for t in range(T):
                G = 0.0
                for i in range(t, min(t+n, T)):
                    state, action, reward, next_state, done = episode[t]
                    G += (self.gamma ** (i-t)) * reward

                if t+n < T:
                    bootstrap, bootstrap_action, _, _, _ = episode[t+n]
                    G += (self.gamma ** n) * self.Q[bootstrap][bootstrap_action]

                state, action, _, _, _ = episode[t]
                td_delta = G - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta

        return self.Q, policy

    def sarsalambda(self, generate_episode: callable, num_episodes: int,
                    lamda: float = 0.9,
                    epsilon: float = 0.1) -> tuple[dict, callable]:
        """SARSA(λ)

        Args:
            generate_episode: Callable that returns (s, a, r, s', done) tuple
            num_episodes: self-evident eh?
            lamda: decay param [0, 1]
            epsilon: Exploration alpha

        Returns:
            Q: Action value function
            π: epsilon-greedy policy
        """
        policy = self.epsilon_greedy_policy(epsilon)
        for i in range(num_episodes):
            if (i+1) % 1000 == 0:
                print(f"\rEpisode {i}/{num_episodes}")

            E = defaultdict(lambda: np.zeros(self.n_actions))
            episode = generate_episode()

            for t in range(len(episode)-1):
                state, action, reward, next_state, done = episode[t]
                next_action_probs = policy(next_state)
                next_action = np.random.choice(self.n_actions,
                                               p=next_action_probs)

                if done:
                    td_target = reward
                else:
                    td_target = reward + (self.gamma *
                                          self.Q[next_state][next_action])
                    td_delta = td_target - self.Q[state][action]

                    E[state][action] += 1.0
                    
                     for s in E:
                         self.Q[s] += self.alpha * td_delta * E[s]
                         E[s] *= self.gamma * lamda

        return self.Q, policy

    def get_policy_state(self, state: int) -> np.ndarray:
        probs = np.zeros(self.n_actions)
        best_action = np.argmax(self.Q[state])
        probs[best_action] = 1.0
        return probs

    def get_q_state(self, state: int) -> np.ndarray:
        return self.Q[state]
