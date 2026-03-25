
"""
File: libql
Use: Q-learning
Update: Sun, 21 Feb 2026
Reference: github/dennybritz
"""

import numpy as np
from collections import defaultdict

class QL:
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

    def q0(self, generate_episode: callable, num_episodes: int,
           epsilon: float = 0.1) -> tuple[dict, callable]:
        """Q-Learning (0)

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
                print(f"\rEpisode {i+1}/{num_episodes}")

            episode = generate_episode()

            for state, action, reward, next_state, done in episode:
                best_next_action = np.argmax(self.Q[next_state])

                if done:
                    td_target = reward
                else:
                    td_target = reward + (self.gamma *
                                          self.Q[next_state][best_next_action])

                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta

        return self.Q, policy

    def qn(self, generate_episode: callable, num_episodes: int, n: int = 5,
           epsilon: float = 0.1) -> tuple[dict, callable]:
        """Q-Learning (N)

        Args:
            generate_episode: Callable that returns (s, a, r, s', done) tuple
            num_episodes: self-evident eh?
            n: n-steps lookahead
            epsilon: Exploration alpha

        Returns:
            Q: Action value function
            π: epsilon-greedy policy
        """
        policy = self.epsilon_greedy_policy(epsilon)

        for i in range(num_episodes):
            if (i+1) % 1000 == 0:
                print(f"\rEpisode {i+1}/{num_episodes}")

            episode = generate_episode()
            T = len(episode)

            for t in range(T):
                G = 0.0

                for i in range(t, min(t+n, T)):
                    _, _, reward, _, _ = episode[i]
                    G += (self.gamma ** (i-t)) * reward

                if t + n < T:
                    bootstrap = episode[t+n][0]
                    best_next_action = np.argmax(self.Q[bootstrap])
                    G += (self.gamma ** n) * self.Q[bootstrap][best_next_action]

                state, action, _, _, _ = episode[t]
                td_delta = G - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta

        return self.Q, policy

    def qlambda(self, generate_episode: callable, num_episodes: int, lamda: float = 0.9,
                epsilon: float = 0.1) -> tuple[dict, callable]:
        """Q-Learning (λ)

        Args:
            generate_episode: Callable that returns (s, a, r, s', done) tuple
            num_episodes: self-evident eh?
            lamda: decay parameter [0, 1]
            epsilon: Exploration alpha

        Returns:
            Q: Action value function
            π: epsilon-greedy policy
        """
        policy = self.epsilon_greedy_policy(epsilon)

        for i in range(num_episodes):
            if (i+1) % 1000 == 0:
                print(f"\rEpisode {i+1}/{num_episodes}")

            E = defaultdict(lambda: np.zeros(self.n_actions))
            episode = generate_episode()

            for state, action, reward, next_state, done in episode:
                best_next_action = np.argmax(self.Q[next_state])

                if done:
                    td_target = reward
                else:
                    td_target = reward + (self.gamma *
                                          self.Q[next_state][best_next_action])

                td_delta = td_target - self.Q[state][action]

                E[state][action] += 1.0

                for s in E:
                    self.Q[s] += self.alpha * td_delta * E[s]
                    E[s] *= self.gamma * lamda

                if action != best_next_action:
                    E = defaultdict(lambda: np.zeros(self.n_actions))

        return self.Q, policy

    def qlambda_forward(self, generate_episode: callable, num_episodes: int,
                        lamda: float = 0.9,
                        epsilon: float = 0.1) -> tuple[dict, callable]:
        """Q(λ) forward view using λ-weighted n-step returns

        Args:
            generate_episode: Callable that returns (s, a, r, s', done) tuple
            num_episodes: self-evident eh?
            lamda: decay parameter [0, 1]
            epsilon: Exploration alpha

        Returns:
            Q: Action value function
            π: epsilon-greedy policy
        """
        policy = self.epsilon_greedy_policy(epsilon)

        for i in range(num_episodes):
            if (i+1) % 1000 == 0:
                print(f"\rEpisode {i+1}/{num_episodes}")

            episode = generate_episode()
            T = len(episode)

            rewards = [ep[2] for ep in episode]
            states  = [ep[0] for ep in episode]
            actions = [ep[1] for ep in episode]

            for t in range(T):
                G_partial = 0.0
                lambda_return = 0.0

                # Accumulate (1-λ) * λ^(n-1) * G_t^n; off-policy greedy bootstrap
                for n in range(1, T - t):
                    G_partial += (self.gamma ** (n - 1)) * rewards[t + n - 1]
                    best_a = np.argmax(self.Q[states[t + n]])
                    G_n = G_partial + (self.gamma ** n) * self.Q[states[t + n]][best_a]
                    lambda_return += (1 - lamda) * (lamda ** (n - 1)) * G_n

                # Terminal return
                G_partial += (self.gamma ** (T - t - 1)) * rewards[T - 1]
                lambda_return += (lamda ** (T - t - 1)) * G_partial

                sa = self.Q[states[t]][actions[t]]
                self.Q[states[t]][actions[t]] += self.alpha * (lambda_return - sa)

        return self.Q, policy

    def qn_backward(self, generate_episode: callable, num_episodes: int,
                    n: int = 5,
                    epsilon: float = 0.1) -> tuple[dict, callable]:
        """Q(n) backward view using eligibility traces with n-step cutoff

        Args:
            generate_episode: Callable that returns (s, a, r, s', done) tuple
            num_episodes: self-evident eh?
            n: trace cutoff in steps
            epsilon: Exploration alpha

        Returns:
            Q: Action value function
            π: epsilon-greedy policy
        """
        policy = self.epsilon_greedy_policy(epsilon)

        for i in range(num_episodes):
            if (i+1) % 1000 == 0:
                print(f"\rEpisode {i+1}/{num_episodes}")

            E = defaultdict(lambda: np.zeros(self.n_actions))
            visit_t = {}
            episode = generate_episode()

            for t, (state, action, reward, next_state, done) in enumerate(episode):
                best_next_action = np.argmax(self.Q[next_state])

                if done:
                    td_target = reward
                else:
                    td_target = reward + (self.gamma * self.Q[next_state][best_next_action])
                td_delta = td_target - self.Q[state][action]

                E[state][action] += 1.0
                visit_t[state] = t

                for s in E:
                    self.Q[s] += self.alpha * td_delta * E[s]
                    E[s] *= self.gamma
                    if (t - visit_t.get(s, t)) >= n:
                        E[s] = np.zeros(self.n_actions)

                # Off-policy correction: zero traces when action deviates from greedy
                if action != best_next_action:
                    E = defaultdict(lambda: np.zeros(self.n_actions))
                    visit_t = {}

        return self.Q, policy

    def get_policy_state(self, state: int) -> np.ndarray:
        probs = np.zeros(self.n_actions)
        best_action = np.argmax(self.Q[state])
        probs[best_action] = 1.0
        return probs

    def get_q(self, state: int) -> np.ndarray:
        return self.Q[state]
