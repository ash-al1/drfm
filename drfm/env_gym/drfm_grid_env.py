"""
File: drfm_grid_env.py
Use: DRFM Gym Environment — Grid World with RF modeling
Update: Wed, 25 Feb 2026

Grid  : G×G cells. Drone moves start→goal. Radar sits at center.
Agent : Single RL agent controls BOTH movement and DRFM technique.

RF equations (env_rf/basic_model.py):
  1. Radar echo   : S = (Pt·Gt·Gr·λ²·σ) / ((4π)³·R⁴·L)
  2. Jam power    : J = (Pj·Gj·Gr·λ²)   / ((4π)²·R²·L)
  3. J/S ratio    : J/S = (Pj·Gj·4π·R²) / (Pt·Gt·σ)
  4. Burn-through : R_BT = sqrt((Pt·Gt·σ)/(Pj·Gj·4π))

Deception model:
  Each DRFM technique injects range and/or velocity error into the
  radar's tracker.  The deception quality modulates Q-factor decay —
  techniques that only overpower (noise jam) degrade Q slowly; those
  that create ghost targets (combined delay + freq-shift) degrade Q
  much faster.

Stochastic radar:
  1. Beam scanning — radar illuminates the drone with probability
     DWELL_PROB each step (TWS phased-array model).  When the beam
     is not on the drone, neither echo nor jam is processed → free miss.
  2. Swerling I RCS — drone cross-section is exponentially distributed
     around SIGMA_MEAN (constant within dwell, independent between
     dwells).  Makes J/S and burn-through range fluctuate naturally.

Track management (IIR):
  - Detection : Q = 1 - F·(1 - Q)                        [toward 1]
  - Miss      : Q *= Q_DECAY * (1 - DW·deception_factor)  [toward 0]
  - Lost      : 3 consecutive misses AND Q < Q_THRESH → bonus + reset
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "env_rf"))

from libs.libql import QL
from basic_model import js_ratio, burn_through
from rf_helpers  import db_to_linear


# ── DRFM technique table ─────────────────────────────────────────────────────
#   Each technique has RF power parameters AND deception quality.
#   range_err : how much false range the retransmit injects      [0–1]
#   vel_err   : how much false velocity (Doppler shift) injected [0–1]
#
#   Deception factor = (range_err + vel_err) / 2
#   → modulates Q-factor decay rate when J/S > 1.

TECHNIQUES = {
    0: {"Pj_dBW": -99.0, "Gj_dBi":  0.0, "range_err": 0.0, "vel_err": 0.0},  # Off
    1: {"Pj_dBW":   5.0, "Gj_dBi":  3.0, "range_err": 0.7, "vel_err": 0.0},  # Delay
    2: {"Pj_dBW":   7.0, "Gj_dBi":  5.0, "range_err": 0.0, "vel_err": 0.7},  # Freq-shift
    3: {"Pj_dBW":  10.0, "Gj_dBi":  7.0, "range_err": 0.8, "vel_err": 0.8},  # Both
}
TECHNIQUE_NAMES = ["Off", "Delay", "Freq-shift", "Both"]
N_TECHNIQUES    = len(TECHNIQUES)

# ── Movement directions ───────────────────────────────────────────────────────
MOVE_DELTAS = {
    0: np.array([ 0,  1]),   # North
    1: np.array([ 0, -1]),   # South
    2: np.array([ 1,  0]),   # East
    3: np.array([-1,  0]),   # West
    4: np.array([ 0,  0]),   # Stay
}
MOVE_NAMES = ["N", "S", "E", "W", "Stay"]
N_MOVES    = len(MOVE_DELTAS)

# ── Composite action helpers ──────────────────────────────────────────────────
N_ACTIONS = N_MOVES * N_TECHNIQUES          # 20


def decode_action(action: int) -> tuple[int, int]:
    """Flat action index → (move_dir, technique_idx)."""
    return action // N_TECHNIQUES, action % N_TECHNIQUES


def encode_action(move: int, tech: int) -> int:
    """(move_dir, technique_idx) → flat action index."""
    return move * N_TECHNIQUES + tech


# ── Environment ───────────────────────────────────────────────────────────────

class DrfmGridEnv(gym.Env):
    """
    DRFM Grid-World Gymnasium environment.

    Observation (MultiDiscrete):
        [drone_x, drone_y, js_bin, q_bin, consec_misses, illuminated]

    Action : Discrete(20)
        Decoded as (move_direction, drfm_technique).
        move  ∈ {N, S, E, W, Stay}   (5)
        tech  ∈ {Off, Delay, Freq, Both} (4)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # ── Radar base parameters ─────────────────────────────────────────────────
    FREQ_HZ    = 10e9
    Pt         = db_to_linear(40.0)     # 40 dBW  transmit power
    Gt         = db_to_linear(30.0)     # 30 dBi  antenna gain
    Gr         = db_to_linear(30.0)     # 30 dBi  receive gain
    SIGMA_MEAN = db_to_linear(0.0)      # 0 dBsm mean drone RCS (1 m² — tactical drone)
    L          = db_to_linear(3.0)      # 3 dB system loss
    LAM        = 3e8 / FREQ_HZ

    # ── Stochastic radar ──────────────────────────────────────────────────────
    #  Beam scanning: TWS phased-array dwell model.
    #  Swerling I:    exponential RCS fluctuation per dwell.
    DWELL_PROB = 0.30       # P(beam on drone) each step

    # ── Deception weight ──────────────────────────────────────────────────────
    #  Controls how much technique quality modulates Q decay.
    #  Q_miss *= Q_DECAY * (1 - DECEPTION_W * deception_factor)
    DECEPTION_W = 0.40

    # ── Track management ──────────────────────────────────────────────────────
    IIR_F      = 0.7        # IIR gain on detection
    Q_DECAY    = 0.75       # base Q decay per miss
    Q_THRESH   = 0.4        # track-lost threshold
    MISS_LIMIT = 3          # consecutive misses to declare lost

    # ── Grid ──────────────────────────────────────────────────────────────────
    GRID   = 10             # cells per side
    CELL_M = 100.0          # metres per cell (100 m → 1 km across full grid)

    # ── Rewards ───────────────────────────────────────────────────────────────
    STEP_RW     =  -0.5     # time pressure — encourages reaching goal  (static)
    TIMEOUT_PEN = -200.0    #                                            (static)
    GOAL_RW     = 100.0     # reached extraction point                   (static)
    DESTROY_PEN = -100.0    # terminal: BT or Q > 0.95                  (static — always fatal)
    MAX_STEPS   =  100

    # Dynamic reward scale factors — see _detect_pen / _lock_pen / _q_bonus below.
    #   DETECT_K  : penalty = -DETECT_K / range_cells  → closer detection hurts more
    #   LOCK_K    : penalty = -LOCK_K * Q_factor        → scales with lock strength
    #   Q_BONUS_K : bonus   = Q_BONUS_K * Q_at_loss    → breaking high-Q track pays more
    DETECT_K    =  40.0
    LOCK_K      =  60.0
    Q_BONUS_K   =  25.0

    # ── Goal-seeking shaping ──────────────────────────────────────────────────
    #  Per-step penalty proportional to remaining distance to goal.
    #  Ensures gradient always points toward goal regardless of radar threat.
    #  Max distance on a 10×10 grid ≈ 12.73 cells → max penalty ≈ -5.0/step.
    DIST_PEN_SCALE = 5.0    # penalty = -DIST_PEN_SCALE * (dist / MAX_DIST)
    MAX_DIST       = math.sqrt(2) * 9  # ≈ 12.73 cells (diagonal of 10×10 grid)
    MOVE_SHAPE_W   = 4.0    # movement shaping coefficient (was 2.0)

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        G = self.GRID
        self.start = np.array([0,     0    ])
        self.goal  = np.array([G - 1, G - 1])
        self.radar = np.array([G // 2, G // 2])

        # obs: [x, y, js_bin, q_bin, consec_misses, illuminated]
        self.observation_space = spaces.MultiDiscrete([G, G, 2, 5, 4, 2])
        self.action_space      = spaces.Discrete(N_ACTIONS)

        self._init_state()

    # ── Internal state helpers ────────────────────────────────────────────────

    def _init_state(self):
        self.drone       = self.start.copy()
        self.Q_factor    = 0.0
        self.consec_misses  = 0
        self.technique   = 0
        self.move_dir    = 4            # Stay
        self.js          = 0.0
        self.steps       = 0
        self.illuminated = False        # current beam status
        self.sigma       = self.SIGMA_MEAN  # current RCS draw

    def _sample_radar(self):
        """Roll stochastic radar state for this step.

        Beam scanning:  Bernoulli trial — is the beam illuminating
                        the drone this dwell?
        Swerling I RCS: Exponential draw around mean cross-section.
                        Constant within dwell, independent across dwells.
        """
        self.illuminated = self.np_random.random() < self.DWELL_PROB
        self.sigma       = self.np_random.exponential(self.SIGMA_MEAN)

    def _range_m(self) -> float:
        diff = (self.drone - self.radar).astype(float)
        return max(np.linalg.norm(diff) * self.CELL_M, 1.0)

    def _range_cells(self) -> float:
        return np.linalg.norm((self.drone - self.radar).astype(float))

    def _bt_range_cells(self, tech_idx: int) -> float:
        """Dynamic burn-through radius (cells) for given technique + current σ."""
        if tech_idx == 0:
            return float("inf")         # no jamming → radar always wins
        tech = TECHNIQUES[tech_idx]
        Pj   = db_to_linear(tech["Pj_dBW"])
        Gj   = db_to_linear(tech["Gj_dBi"])
        r_bt = burn_through(self.Pt, self.Gt, self.sigma, Pj, Gj)
        return r_bt / self.CELL_M

    def _in_bt_zone(self, tech_idx: int) -> bool:
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
        """Combined deception quality in [0, 1]."""
        t = TECHNIQUES[tech_idx]
        return (t["range_err"] + t["vel_err"]) / 2.0

    def _detect_pen(self) -> float:
        """Detection penalty — inverse of range (closer = worse).

        At 1 cell  : -DETECT_K      (maximum, very close)
        At 5 cells : -DETECT_K / 5  (moderate)
        At 10 cells: -DETECT_K / 10 (mild, distant detection)
        """
        return -self.DETECT_K / max(self._range_cells(), 0.5)

    def _lock_pen(self) -> float:
        """Lock penalty — linear in Q_factor (partially locked = partial penalty).

        Replaces the cliff at Q > 0.85.  Penalty grows continuously from 0
        toward -LOCK_K as Q approaches 1.0.
        """
        return -self.LOCK_K * self.Q_factor

    def _q_bonus(self, q_before: float) -> float:
        """Track-broken bonus — proportional to the Q that was just wiped.

        Breaking a near-certain track (Q≈0.9) is worth far more than
        breaking a nascent one (Q≈0.1).
        """
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
        """Apply agent-chosen movement, clipped to grid bounds."""
        new_pos = self.drone + MOVE_DELTAS[move_dir]
        new_pos = np.clip(new_pos, 0, self.GRID - 1)
        self.drone = new_pos

    # ── Gym interface ─────────────────────────────────────────────────────────

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

        # ── Movement first ────────────────────────────────────────────────────
        self._move_drone(move_dir)

        dist_after = np.linalg.norm((self.drone - self.goal).astype(float))
        # Movement shaping: reward closing on goal, penalise drifting away.
        # Distance penalty: always-on pull toward goal (overrides hide-in-place).
        reward = (self.STEP_RW
                  + self.MOVE_SHAPE_W * (dist_before - dist_after)
                  - self.DIST_PEN_SCALE * (dist_after / self.MAX_DIST))

        # ── Sample stochastic radar for this dwell ────────────────────────────
        self._sample_radar()

        # ── RF computation ────────────────────────────────────────────────────
        self.js    = self._compute_js(tech_idx)
        in_bt      = self._in_bt_zone(tech_idx)
        terminated = False

        # ── Beam not on drone → free miss (scan gap) ─────────────────────────
        if not self.illuminated:
            # Radar not looking — Q decays at base rate, no deception bonus
            q_before        = self.Q_factor
            self.Q_factor  *= self.Q_DECAY
            self.consec_misses += 1
            if self.consec_misses >= self.MISS_LIMIT and self.Q_factor < self.Q_THRESH:
                reward         += self._q_bonus(q_before)
                self.consec_misses = 0

        # ── Burn-through zone → instant lock + destruction ────────────────────
        elif in_bt:
            self.Q_factor   = 1.0
            self.consec_misses = 0
            # Use dynamic detect + lock penalties at current range/Q, plus static destroy
            reward         += self._detect_pen() + self._lock_pen() + self.DESTROY_PEN
            terminated      = True

        # ── Illuminated: J/S contest ──────────────────────────────────────────
        else:
            jam_wins = self.js > 1.0

            if jam_wins:
                # Jammer overpowers echo — technique deception modulates decay
                q_before        = self.Q_factor
                df              = self._deception_factor(tech_idx)
                rate            = self.Q_DECAY * (1.0 - self.DECEPTION_W * df)
                self.Q_factor  *= rate
                self.consec_misses += 1
                if self.consec_misses >= self.MISS_LIMIT and self.Q_factor < self.Q_THRESH:
                    reward         += self._q_bonus(q_before)
                    self.consec_misses = 0
            else:
                # Radar detects: IIR update — penalty scales with range and new Q
                self.Q_factor   = 1.0 - self.IIR_F * (1.0 - self.Q_factor)
                self.consec_misses = 0
                reward         += self._detect_pen() + self._lock_pen()
                if self.Q_factor > 0.95:
                    reward     += self.DESTROY_PEN
                    terminated  = True

        # ── Goal check ────────────────────────────────────────────────────────
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

    # ── Rendering ─────────────────────────────────────────────────────────────

    # ANSI colour helpers — only used in render output
    _RED  = "\033[91m"   # bright red   — radar / beam-on drone / BT zone
    _GRN  = "\033[92m"   # bright green — goal / beam-off drone / jammer winning
    _YLW  = "\033[93m"   # bright yellow — BT zone cells
    _CYN  = "\033[96m"   # bright cyan  — radar cell
    _RST  = "\033[0m"    # reset

    def _build_frame(self) -> str:
        G = self.GRID

        # dynamic burn-through radius for display
        bt_cells = self._bt_range_cells(self.technique)

        # ── build two grids: plain (for width calc) and coloured (for display) ─
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

        # Drone — red when beam is on (illuminated), green during scan gap
        drone_c = self._RED if self.illuminated else self._GRN
        plain [self.drone[1]][self.drone[0]] = "D"
        colour[self.drone[1]][self.drone[0]] = f"{drone_c}D{self._RST}"

        x_axis = f"   {' '.join(str(x) for x in range(G))}"
        grid_plain  = [f"{y:2d} {' '.join(plain[y])}"  for y in range(G - 1, -1, -1)]
        grid_colour = [f"{y:2d} {' '.join(colour[y])}" for y in range(G - 1, -1, -1)]
        grid_plain .append(x_axis)
        grid_colour.append(x_axis)

        # ── info panel (plain text — drives box width) ────────────────────────
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
            f" Consec radar misses: {self.consec_misses}/{self.MISS_LIMIT}"
            f"   Range: {self._range_m()/1e3:.2f} km",
            f" σ: {sig_db:+.1f} dBsm   BT: {bt_str}",
            "",
        ]

        # Coloured versions of lines that need it
        beam_tag_c = (f"{self._RED}BEAM ON {self._RST}" if self.illuminated
                      else f"{self._GRN}scan gap{self._RST}")
        js_tag_c   = (f"{self._GRN}{js_tag}{self._RST}" if jam_ok
                      else f"{self._RED}{js_tag}{self._RST}")

        info_colour = list(info_plain)  # copy; only patch the two coloured lines
        info_colour[2] = (f" J/S:  {js_db:+.1f} dB  [{js_tag_c}]   [{beam_tag_c}]")

        # ── box it — widths driven by plain text only ─────────────────────────
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


# ── Q-learning training ───────────────────────────────────────────────────────

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
    """Replay trained greedy policy with terminal animation."""
    env = DrfmGridEnv(render_mode="human")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state  = tuple(obs)
        total  = 0.0

        print("\033[2J\033[H", end="", flush=True)
        print(f"[ Trained agent — episode {ep}/{num_episodes} ]")
        env.render()
        time.sleep(step_delay * 2)

        while True:
            action = int(np.argmax(ql.Q[state]))
            move_dir, tech_idx = decode_action(action)

            obs, reward, terminated, truncated, info = env.step(action)
            state  = tuple(obs)
            total += reward

            print("\033[2J\033[H", end="", flush=True)
            print(f"[ Trained agent — ep {ep}/{num_episodes}  rw: {total:+.1f} ]")
            env.render()
            time.sleep(step_delay)

            if terminated or truncated:
                if np.array_equal(env.drone, env.goal):
                    result = "REACHED GOAL"
                elif env.Q_factor >= 0.95:
                    result = "DESTROYED  (missile lock)"
                elif info["in_bt"]:
                    result = "DESTROYED  (burn-through)"
                else:
                    result = "TIMEOUT"
                print(f"\n  >> {result} <<")
                time.sleep(1.5)
                break

    env.close()


if __name__ == "__main__":
    ql = train(num_episodes=20_000, render_final=True)
