"""
File: drfm_grid_env.py
Use: Intuition builder using Gymnasium for DRFM module
Update: Wed, 25 Feb 2026

RF equations (basic_model.py):
- Echo  = (Pt·Gt·Gr·λ²·σ) / ((4π)³·R⁴·L)
- Jam_p = (Pj·Gj·Gr·λ²)   / ((4π)²·R²·L)
- JSR   = (Pj·Gj·4π·R²)   / (Pt·Gt·σ)
- Burn  = sqrt((Pt·Gt·σ)  / (Pj·Gj·4π))
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from drfm.algorithms.QLearn import QL
from drfm.envs.gym.basic_model import js_ratio, burn_through
from drfm.envs.gym.rf_helpers  import db_to_linear


# DRFM Actions
TECHNIQUES = {
    0: {"Pj_dBW": -99.0, "Gj_dBi":  0.0, "range_err": 0.0, "vel_err": 0.0},  # Off
    1: {"Pj_dBW":   5.0, "Gj_dBi":  3.0, "range_err": 0.7, "vel_err": 0.0},  # Delay
    2: {"Pj_dBW":   7.0, "Gj_dBi":  5.0, "range_err": 0.0, "vel_err": 0.7},  # Freq
    3: {"Pj_dBW":  10.0, "Gj_dBi":  7.0, "range_err": 0.8, "vel_err": 0.8},  # Both
}
TECHNIQUE_NAMES = ["Off", "Delay", "Freq", "Both"]
N_TECHNIQUES    = len(TECHNIQUES)

# Movement
MOVE_DELTAS = {
    0: np.array([ 0,  1]),   # North
    1: np.array([ 0, -1]),   # South
    2: np.array([ 1,  0]),   # East
    3: np.array([-1,  0]),   # West
    4: np.array([ 0,  0]),   # Stay
}
MOVE_NAMES = ["N", "S", "E", "W", "Stay"]
N_MOVES    = len(MOVE_DELTAS)

N_ACTIONS = N_MOVES * N_TECHNIQUES

def decode_action(action: int) -> tuple[int, int]:
    return action // N_TECHNIQUES, action % N_TECHNIQUES

def encode_action(move: int, tech: int) -> int:
    return move * N_TECHNIQUES + tech

class DrfmGridEnv(gym.Env):
    """DRFM Grid-World Gymnasium environment

    Obs:
        [drone_x, drone_y, js_bin, q_bin, misses, illuminated]

    Action:
        move : {N, S, E, W, Stay}
        tech : {Off, Delay, Freq, Both}
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Radar
    FREQ_HZ    = 10e9
    Pt         = db_to_linear(40.0)
    Gt         = db_to_linear(30.0)
    Gr         = db_to_linear(30.0)
    SIGMA_MEAN = db_to_linear(0.0)
    L          = db_to_linear(3.0)
    LAM        = 3e8 / FREQ_HZ
    DWELL_PROB = 0.30
    DECEPTION_W = 0.40
    IIR_F      = 0.7
    Q_DECAY    = 0.75
    Q_THRESH   = 0.4
    MISS_LIMIT = 3

    # Gridworld
    GRID   = 10
    CELL_M = 100.0 # Range per cell

    # Rewards
    STEP_RW     =  -0.5
    TIMEOUT_PEN = -200.0
    GOAL_RW     = 100.0
    DESTROY_PEN = -100.0
    MAX_STEPS   =  100

    # Penalties, discounts
    DETECT_K    =  40.0
    LOCK_K      =  60.0
    Q_BONUS_K   =  25.0
    DIST_PEN_SCALE = 5.0
    MAX_DIST       = math.sqrt(2) * 9
    MOVE_SHAPE_W   = 4.0

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        G = self.GRID
        self.start = np.array([0,     0    ])
        self.goal  = np.array([G - 1, G - 1])
        self.radar = np.array([G // 2, G // 2])

        # obs: [x, y, js_bin, q_bin, misses, illuminated]
        self.observation_space = spaces.MultiDiscrete([G, G, 2, 5, 4, 2])
        self.action_space      = spaces.Discrete(N_ACTIONS)

        self._init_state()

    def _init_state(self):
        self.drone       = self.start.copy()
        self.Q_factor    = 0.0
        self.consec_misses  = 0
        self.technique   = 0
        self.move_dir    = 4
        self.js          = 0.0
        self.steps       = 0
        self.illuminated = False
        self.sigma       = self.SIGMA_MEAN

    def _sample_radar(self):
        """Radar is stochastic

        Uses Beam scanning, Swerling I
        """
        self.illuminated = self.np_random.random() < self.DWELL_PROB
        self.sigma       = self.np_random.exponential(self.SIGMA_MEAN)

    def _range_m(self) -> float:
        diff = (self.drone - self.radar).astype(float)
        return max(np.linalg.norm(diff) * self.CELL_M, 1.0)

    def _range_cells(self) -> float:
        return np.linalg.norm((self.drone - self.radar).astype(float))

    def _bt_range_cells(self, tech_idx: int) -> float:
        """Burn through cells based off RCS"""
        if tech_idx == 0:
            return float("inf")
        tech = TECHNIQUES[tech_idx]
        Pj   = db_to_linear(tech["Pj_dBW"])
        Gj   = db_to_linear(tech["Gj_dBi"])
        r_bt = burn_through(self.Pt, self.Gt, self.sigma, Pj, Gj)
        return r_bt / self.CELL_M

    def _in_bt_zone(self, tech_idx: int) -> bool:
        """Terminate if burn drone is too close to radar"""
        return self._range_cells() <= self._bt_range_cells(tech_idx)

    def _compute_js(self, tech_idx: int) -> float:
        if tech_idx == 0:
            return 0.0
        tech = TECHNIQUES[tech_idx]
        Pj   = db_to_linear(tech["Pj_dBW"])
        Gj   = db_to_linear(tech["Gj_dBi"])
        return js_ratio(Pj, Gj, self.Pt, self.Gt, self.sigma, self._range_m())

    @staticmethod
    def _deception_factor(tech_idx: int) -> float:
        t = TECHNIQUES[tech_idx]
        return (t["range_err"] + t["vel_err"]) / 2.0

    def _detect_pen(self) -> float:
        return -self.DETECT_K / max(self._range_cells(), 0.5)

    def _lock_pen(self) -> float:
        return -self.LOCK_K * self.Q_factor

    def _q_bonus(self, q_before: float) -> float:
        return self.Q_BONUS_K * q_before

    def _q_bin(self) -> int:
        return min(int(self.Q_factor * 5), 4)

    def _obs(self) -> np.ndarray:
        return np.array([
            self.drone[0],
            self.drone[1],
            1 if (self.js > 1.0 and not self._in_bt_zone(self.technique)) else 0,
            self._q_bin(),
            min(self.consec_misses, 3),
            1 if self.illuminated else 0,
        ], dtype=np.int64)

    def _move_drone(self, move_dir: int):
        new_pos = self.drone + MOVE_DELTAS[move_dir]
        new_pos = np.clip(new_pos, 0, self.GRID - 1)
        self.drone = new_pos

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        self._sample_radar()
        self.js = self._compute_js(self.technique)
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        move_dir, tech_idx = decode_action(action)

        self.steps     += 1
        self.technique  = tech_idx
        self.move_dir   = move_dir

        dist_before = np.linalg.norm((self.drone - self.goal).astype(float))
        self._move_drone(move_dir)

        dist_after = np.linalg.norm((self.drone - self.goal).astype(float))
        reward = (self.STEP_RW
                  + self.MOVE_SHAPE_W * (dist_before - dist_after)
                  - self.DIST_PEN_SCALE * (dist_after / self.MAX_DIST))
        self._sample_radar()

        # ~RF simulation
        self.js    = self._compute_js(tech_idx)
        in_bt      = self._in_bt_zone(tech_idx)
        terminated = False

        # Scanner not on drone yet
        if not self.illuminated:
            q_before        = self.Q_factor
            self.Q_factor  *= self.Q_DECAY
            self.consec_misses += 1
            if self.consec_misses >= self.MISS_LIMIT and self.Q_factor < self.Q_THRESH:
                reward         += self._q_bonus(q_before)
                self.consec_misses = 0

        # Terminate if drone is in burn through range
        elif in_bt:
            self.Q_factor   = 1.0
            self.consec_misses = 0
            reward         += self._detect_pen() + self._lock_pen() + self.DESTROY_PEN
            terminated      = True

        # Contest JSR
        else:
            jam_wins = self.js > 1.0

            if jam_wins:
                q_before        = self.Q_factor
                df              = self._deception_factor(tech_idx)
                rate            = self.Q_DECAY * (1.0 - self.DECEPTION_W * df)
                self.Q_factor  *= rate
                self.consec_misses += 1
                if self.consec_misses >= self.MISS_LIMIT and self.Q_factor < self.Q_THRESH:
                    reward         += self._q_bonus(q_before)
                    self.consec_misses = 0
            else:
                self.Q_factor   = 1.0 - self.IIR_F * (1.0 - self.Q_factor)
                self.consec_misses = 0
                reward         += self._detect_pen() + self._lock_pen()
                if self.Q_factor > 0.95:
                    reward     += self.DESTROY_PEN
                    terminated  = True

        # Check if drone has reached goal
        if not terminated and np.array_equal(self.drone, self.goal):
            reward    += self.GOAL_RW
            terminated = True

        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated:
            reward += self.TIMEOUT_PEN

        info = {
            "js":           self.js,
            "Q_factor":     self.Q_factor,
            "detected":     self.illuminated and (self.js <= 1.0) and not in_bt,
            "in_bt":        in_bt,
            "illuminated":  self.illuminated,
            "technique":    TECHNIQUE_NAMES[tech_idx],
            "move":         MOVE_NAMES[move_dir],
            "range_m":      self._range_m(),
            "sigma_dBsm":   10.0 * math.log10(max(self.sigma, 1e-30)),
            "bt_cells":     self._bt_range_cells(tech_idx),
        }
        return self._obs(), reward, terminated, truncated, info

    # ANSI color better intuition of whats going on
    _RED  = "\033[91m"
    _GRN  = "\033[92m"
    _YLW  = "\033[93m"
    _CYN  = "\033[96m"
    _RST  = "\033[0m"

    def _build_frame(self) -> str:
        G = self.GRID

        # Burn through range changes overtime (non-deterministically)
        bt_cells = self._bt_range_cells(self.technique)

        # Build grids
        plain  = [["." for _ in range(G)] for _ in range(G)]
        colour = [["." for _ in range(G)] for _ in range(G)]

        for cy in range(G):
            for cx in range(G):
                dist = math.hypot(cx - self.radar[0], cy - self.radar[1])
                if dist <= bt_cells:
                    plain[cy][cx]  = "~"
                    colour[cy][cx] = f"{self._YLW}~{self._RST}"

        # Radar
        plain [self.radar[1]][self.radar[0]] = "R"
        colour[self.radar[1]][self.radar[0]] = f"{self._CYN}R{self._RST}"

        # Goal
        plain [self.goal[1]][self.goal[0]] = "G"
        colour[self.goal[1]][self.goal[0]] = f"{self._GRN}G{self._RST}"

        # Drone 
        # If red then its illuminated, green if gap
        drone_c = self._RED if self.illuminated else self._GRN
        plain [self.drone[1]][self.drone[0]] = "D"
        colour[self.drone[1]][self.drone[0]] = f"{drone_c}D{self._RST}"

        x_axis = f"   {' '.join(str(x) for x in range(G))}"
        grid_plain  = [f"{y:2d} {' '.join(plain[y])}"  for y in range(G - 1, -1, -1)]
        grid_colour = [f"{y:2d} {' '.join(colour[y])}" for y in range(G - 1, -1, -1)]
        grid_plain .append(x_axis)
        grid_colour.append(x_axis)

        in_bt    = self._in_bt_zone(self.technique)
        jam_ok   = self.js > 1.0 and not in_bt
        js_db    = 10.0 * math.log10(max(self.js, 1e-12))
        js_tag   = ("JAM" if jam_ok else "RAD") + (" [BT]" if in_bt else "")
        beam_tag = "BEAM ON " if self.illuminated else "scan gap"
        q_n      = int(self.Q_factor * 10)
        q_bar    = "#" * q_n + "-" * (10 - q_n)
        lock     = " !! LOCK"    if self.Q_factor > 0.95 else \
                   " ! locking"  if self.Q_factor > 0.85 else ""
        sig_db   = 10.0 * math.log10(max(self.sigma, 1e-30))
        bt_str   = f"{bt_cells:.1f} cells" if self.technique > 0 else "n/a (off)"

        info_plain = [
            "",
            f" Step: {self.steps:<4d}  Move: {MOVE_NAMES[self.move_dir]:<4s}"
            f"  Tech: {TECHNIQUE_NAMES[self.technique]}",
            f" J/S:  {js_db:+.1f} dB  [{js_tag}]   [{beam_tag}]",
            f" Track:[{q_bar}] {self.Q_factor:.2f}{lock}",
            f"   Range: {self._range_m()/1e3:.2f} km",
            f" σ: {sig_db:+.1f} dBsm   BT: {bt_str}",
            "",
        ]

        beam_tag_c = (f"{self._RED}BEAM ON {self._RST}" if self.illuminated
                      else f"{self._GRN}scan gap{self._RST}")
        js_tag_c   = (f"{self._GRN}{js_tag}{self._RST}" if jam_ok
                      else f"{self._RED}{js_tag}{self._RST}")

        info_colour = list(info_plain)  # copy; only patch the two coloured lines
        info_colour[2] = (f" J/S:  {js_db:+.1f} dB  [{js_tag_c}]   [{beam_tag_c}]")

        all_plain = grid_plain + info_plain
        IW        = max(len(l) for l in all_plain) + 2
        border    = "+" + "-" * IW + "+"
        sep       = "|" + "-" * IW + "|"

        frame = [border]
        for plain_l, colour_l in zip(grid_plain, grid_colour):
            pad = IW - 2 - len(plain_l)
            frame.append(f"| {colour_l}{' ' * pad} |")
        frame.append(sep)
        for plain_l, colour_l in zip(info_plain, info_colour):
            pad = IW - 2 - len(plain_l)
            frame.append(f"| {colour_l}{' ' * pad} |")
        frame.append(border)

        return "\n".join(frame)

    def render(self):
        frame = self._build_frame()
        if self.render_mode == "human":
            print(frame)
        return frame

def train(
    num_episodes: int   = 50_000,
    gamma:        float = 0.99,
    alpha:        float = 0.2,
    eps_start:    float = 1.0,
    eps_end:      float = 0.05,
    eps_decay:    float = 0.9995,
    render_final: bool  = True,
) -> QL:
    env = DrfmGridEnv()
    ql  = QL(n_states=0, n_actions=env.action_space.n, gamma=gamma, alpha=alpha)
    eps = eps_start
    ep_rewards = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state  = tuple(obs)
        total  = 0.0

        while True:
            policy = ql.epsilon_greedy_policy(eps)
            action = int(np.random.choice(env.action_space.n, p=policy(state)))

            obs_next, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(obs_next)
            done       = terminated or truncated

            best_next = float(np.max(ql.Q[next_state]))
            td_target = reward + (0.0 if done else gamma * best_next)
            ql.Q[state][action] += alpha * (td_target - ql.Q[state][action])

            state  = next_state
            total += reward
            if done:
                break

        eps = max(eps_end, eps * eps_decay)
        ep_rewards.append(total)

        if ep % 1000 == 0:
            avg = np.mean(ep_rewards[-1000:])
            print(f"  ep {ep:>6}/{num_episodes}  avg_rw={avg:+.1f}  eps={eps:.3f}")

    env.close()

    if render_final:
        _run_greedy(ql)

    return ql


def _run_greedy(ql: QL, num_episodes: int = 5, step_delay: float = 0.3):
    """Visualization after training"""
    env = DrfmGridEnv(render_mode="human")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state  = tuple(obs)
        total  = 0.0

        print("\033[2J\033[H", end="", flush=True)
        print(f"[Agent episode {ep}/{num_episodes}]")
        env.render()
        time.sleep(step_delay * 2)

        while True:
            action = int(np.argmax(ql.Q[state]))
            move_dir, tech_idx = decode_action(action)

            obs, reward, terminated, truncated, info = env.step(action)
            state  = tuple(obs)
            total += reward

            print("\033[2J\033[H", end="", flush=True)
            print(f"[Ep {ep}/{num_episodes}  rw: {total:+.1f}]")
            env.render()
            time.sleep(step_delay)

            if terminated or truncated:
                if np.array_equal(env.drone, env.goal):
                    result = "Terminated"
                elif env.Q_factor >= 0.95:
                    result = "Hit"
                elif info["in_bt"]:
                    result = "Burn through"
                else:
                    result = "TIMEOUT"
                print(f"\n  {result}")
                time.sleep(1.5)
                break

    env.close()


if __name__ == "__main__":
    ql = train(num_episodes=5_000, render_final=True)
