# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali

from __future__ import annotations

import logging
import math

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.utils import configclass

from dynamics import Allocation, Motor
from .mdp.radar import RadarManager
from .mdp.drfm_action import POR_MIN, POR_MAX, VPR_MIN, VPR_MAX, POWER_COST
from .drfm_env import (
    DroneReconSceneCfgDRFM,
    _drone_slot,
    _OBSTACLE_FOOTPRINTS,
    _OBSTACLE_GEOM,
    _RADAR_POSITIONS,
    _RADAR_EXCLUSION_ZONES,
)

logger = logging.getLogger(__name__)

_N_AGENTS = 5

def _robot_key(i: int) -> str:
    return "robot" if i == 0 else f"robot_{i}"

def _sensor_key(i: int) -> str:
    return "collision_sensor" if i == 0 else f"collision_sensor_{i}"

_AGENTS = [f"drone_{i}" for i in range(_N_AGENTS)]
_ROBOT_KEYS = {f"drone_{i}": _robot_key(i) for i in range(_N_AGENTS)}
_SENSOR_KEYS = {f"drone_{i}": _sensor_key(i) for i in range(_N_AGENTS)}

_OBS_DIM = 56   # per agent: 3+1+4+1+1+3+3+32+8
_ACT_DIM = 11   # per agent: 4 control + 7 DRFM

# Waypoint sampling config (matches CommandsCfg)
_WP_PER_EP = 3
_ARRIVAL_THRESH = 1.0
_GOAL_X = (44.0, 48.0)
_GOAL_Y = (-10.0, 10.0)
_GOAL_Z = (1.0, 3.0)
_TARGET_Z = 3.0

# Reward weights (matches RewardsCfg)
_W = {
    "progress": 5.0,
    "forward_speed": 2.0,
    "heading": 2.0,
    "waypoint_reached": 50.0,
    "completion_bonus": 100.0,
    "upright": 1.0,
    "altitude_band": -5.0,
    "terminating": -200.0,
    "step_penalty": -0.01,
    "ang_vel_l2": -0.02,
    "action_smooth": -0.01,
    "proximity": -3.0,
    "illumination": -2.0,
    "power_conserve": 0.5,
    "drfm_effective": 2.0,
    "smart_jam": 1.0,
}

# ControlActionCfg defaults
_ARM_LEN = 0.035
_THRUST_COEF = 2.25e-7
_DRAG_COEF = 1.5e-9
_OMEGA_MAX = 5145.0
_HOVER_FRAC = 0.25
_TAUS = (0.0001, 0.0001, 0.0001, 0.0001)
_INIT_OMEGA = (2572.5, 2572.5, 2572.5, 2572.5)
_MAX_RATE = (50000.0,) * 4
_MIN_RATE = (-50000.0,) * 4


def _denorm(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (x.clamp(-1.0, 1.0) + 1.0) * 0.5 * (hi - lo)


class WaypointTracker:
    """Standalone waypoint tracker — ported from WaypointCommand without CommandTerm machinery."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        waypoints_per_episode: int = _WP_PER_EP,
        arrival_threshold: float = _ARRIVAL_THRESH,
        goal_x_range: tuple = _GOAL_X,
        goal_y_range: tuple = _GOAL_Y,
        goal_z_range: tuple = _GOAL_Z,
        exclusion_zones: tuple = _RADAR_EXCLUSION_ZONES,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.waypoints_per_episode = waypoints_per_episode
        self.arrival_threshold = arrival_threshold
        self.goal_x_range = goal_x_range
        self.goal_y_range = goal_y_range
        self.goal_z_range = goal_z_range
        self.exclusion_zones = exclusion_zones

        self._command = torch.zeros(num_envs, 3, device=device)
        self._waypoints = torch.zeros(num_envs, waypoints_per_episode, 3, device=device)
        self._waypoint_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._waypoints_visited = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.just_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._all_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._previous_pos = torch.zeros(num_envs, 3, device=device)

    @property
    def command(self) -> torch.Tensor:
        """Current waypoint position in world frame [N, 3]."""
        return self._command

    @property
    def waypoints_remaining(self) -> torch.Tensor:
        """Count of unvisited waypoints [N]."""
        return (self.waypoints_per_episode - self._waypoints_visited).clamp(min=0).float()

    @property
    def all_done(self) -> torch.Tensor:
        """True for envs that have visited all waypoints [N]."""
        return self._all_done

    def reset(self, env_ids: torch.Tensor, drone_pos_w: torch.Tensor, env_origins: torch.Tensor) -> None:
        """Resample waypoints for env_ids and record current drone position as previous."""
        n = len(env_ids)
        wp = torch.zeros(n, self.waypoints_per_episode, 3, device=self.device)
        for i in range(self.waypoints_per_episode):
            wp[:, i, :] = self._sample_valid_positions(n) + env_origins[env_ids]
        self._waypoints[env_ids] = wp
        self._waypoint_idx[env_ids] = 0
        self._waypoints_visited[env_ids] = 0
        self.just_reached[env_ids] = False
        self._all_done[env_ids] = False
        self._command[env_ids] = wp[:, 0, :]
        self._previous_pos[env_ids] = drone_pos_w[env_ids]

    def update(self, drone_pos_w: torch.Tensor) -> None:
        """Advance to the next waypoint when within arrival_threshold."""
        self.just_reached[:] = False
        idx = torch.arange(self.num_envs, device=self.device)
        current_goals = self._waypoints[idx, self._waypoint_idx]
        dist = torch.norm(drone_pos_w - current_goals, dim=1)
        arrived_mask = (dist < self.arrival_threshold) & ~self._all_done

        if arrived_mask.any():
            self.just_reached |= arrived_mask
            self._waypoint_idx[arrived_mask] += 1
            self._waypoints_visited[arrived_mask] += 1
            newly_done = self._waypoint_idx >= self.waypoints_per_episode
            self._all_done |= newly_done
            advanced = arrived_mask & ~newly_done
            if advanced.any():
                adv_wp = self._waypoint_idx[advanced]
                self._command[advanced] = self._waypoints[idx[advanced], adv_wp]

        self._previous_pos = drone_pos_w.clone()

    def _sample_valid_positions(self, n: int) -> torch.Tensor:
        ranges = [self.goal_x_range, self.goal_y_range, self.goal_z_range]
        pos = torch.stack(
            [torch.empty(n, device=self.device).uniform_(*r) for r in ranges], dim=1
        )
        valid = self._is_clear(pos)
        for _ in range(10):
            if valid.all():
                break
            count = int((~valid).sum().item())
            new_pos = torch.stack(
                [torch.empty(count, device=self.device).uniform_(*r) for r in ranges], dim=1
            )
            pos[~valid] = new_pos
            valid = self._is_clear(pos)
        return pos

    def _is_clear(self, pos: torch.Tensor) -> torch.Tensor:
        clear = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
        for rx, ry, radius in self.exclusion_zones:
            dist_2d = torch.sqrt((pos[:, 0] - rx) ** 2 + (pos[:, 1] - ry) ** 2)
            clear &= dist_2d >= radius
        return clear


@configclass
class DroneMARLSceneCfg(DroneReconSceneCfgDRFM):
    """Scene with 5 controllable drones."""

    robot_2, collision_sensor_2 = _drone_slot(2)
    robot_3, collision_sensor_3 = _drone_slot(3)
    robot_4, collision_sensor_4 = _drone_slot(4)


@configclass
class DroneMARLEnvCfg(DirectMARLEnvCfg):
    """Config for the five-drone cooperative DRFM environment."""

    possible_agents: list = _AGENTS
    observation_spaces: dict = {a: _OBS_DIM for a in _AGENTS}
    action_spaces: dict = {a: _ACT_DIM for a in _AGENTS}
    state_space: int = -1  # auto-concatenate all agent obs for centralized critic

    scene: DroneMARLSceneCfg = DroneMARLSceneCfg(num_envs=2048, env_spacing=70.0)

    decimation: int = 8
    episode_length_s: float = 25.0

    def __post_init__(self) -> None:
        self.viewer.eye = (-5.0, 0.0, 20.0)
        self.viewer.lookat = (25.0, 0.0, 1.0)
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.min_velocity_iteration_count = 1


class DroneMARLEnv(DirectMARLEnv):
    """Two-drone cooperative EW environment using the direct MARL workflow.

    Both drones share the same observation/action space and are trained with MAPPO
    (shared policy, centralized critic). Each drone independently manages its own
    RadarManager and DRFM state.

    TODO: Option B — a single shared RadarManager that tracks the nearest/most-threatening
    drone would be more physically realistic. Currently using Option A (independent managers).
    """

    cfg: DroneMARLEnvCfg

    def _setup_scene(self) -> None:
        """Initialize per-agent dynamics, radar, and waypoint objects."""
        n = self.num_envs
        d = self.device
        physics_dt = self.physics_dt

        # Scene asset references
        self._robots = {a: self.scene[_ROBOT_KEYS[a]] for a in _AGENTS}
        self._sensors = {a: self.scene[_SENSOR_KEYS[a]] for a in _AGENTS}
        self._body_ids: dict | None = None  # resolved after sim.reset() on first _apply_action

        # Per-agent flight dynamics
        self._allocation = {
            a: Allocation(n, _ARM_LEN, _THRUST_COEF, _DRAG_COEF, d, torch.float32)
            for a in _AGENTS
        }
        self._motor = {
            a: Motor(n, _TAUS, _INIT_OMEGA, _MAX_RATE, _MIN_RATE, physics_dt, use=False, device=d)
            for a in _AGENTS
        }

        # Per-agent cached thrust/moment for apply_action
        self._thrust = {a: torch.zeros(n, 1, 3, device=d) for a in _AGENTS}
        self._moment = {a: torch.zeros(n, 1, 3, device=d) for a in _AGENTS}

        # Per-agent DRFM electronic warfare state
        self._technique = {a: torch.zeros(n, dtype=torch.long, device=d) for a in _AGENTS}
        self._prev_technique = {a: torch.zeros(n, dtype=torch.long, device=d) for a in _AGENTS}
        self._pull_off_rate = {a: torch.zeros(n, device=d) for a in _AGENTS}
        self._vel_pull_off_rate = {a: torch.zeros(n, device=d) for a in _AGENTS}
        self._coord = {a: torch.zeros(n, device=d) for a in _AGENTS}
        self._power = {a: torch.ones(n, device=d) for a in _AGENTS}
        self._raw_control = {a: torch.zeros(n, 4, device=d) for a in _AGENTS}

        # Power cost tensor for fast indexing
        self._power_cost = {a: torch.tensor(POWER_COST, device=d) for a in _AGENTS}

        # Per-agent radar state machines (Option A: independent)
        self._radar_managers = {
            a: RadarManager(n, d, _RADAR_POSITIONS, _OBSTACLE_GEOM)
            for a in _AGENTS
        }

        # Per-agent waypoint trackers
        self._waypoints = {
            a: WaypointTracker(n, d)
            for a in _AGENTS
        }

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Process actions: update DRFM state and compute thrust/moment for each agent."""
        dt = float(self.step_dt)
        for a in _AGENTS:
            act = torch.nan_to_num(actions[a], nan=0.0, posinf=1.0, neginf=-1.0)
            ctrl = act[:, :4]
            drfm = act[:, 4:11]

            # -- control: map [-1,1] to omega -> thrust/moment
            self._raw_control[a] = ctrl.clamp(-1.0, 1.0)
            hf = _HOVER_FRAC
            mapped = torch.where(
                self._raw_control[a] < 0,
                hf * (self._raw_control[a] + 1.0),
                hf + (1.0 - hf) * self._raw_control[a],
            ).clamp(min=hf)
            omega_ref = _OMEGA_MAX * torch.sqrt(mapped)
            omega_real = self._motor[a].compute(omega_ref)
            proc = self._allocation[a].compute(omega_real)
            self._thrust[a][:, 0, 2] = proc[:, 0]
            self._moment[a][:, 0, :] = proc[:, 1:]

            # -- DRFM: update technique, denorm params, call radar
            self._prev_technique[a][:] = self._technique[a]
            technique = drfm[:, :4].argmax(dim=1).long()
            technique = torch.where(self._power[a] <= 0.0, torch.zeros_like(technique), technique)
            self._technique[a][:] = technique
            self._pull_off_rate[a] = _denorm(drfm[:, 4], POR_MIN, POR_MAX)
            self._vel_pull_off_rate[a] = _denorm(drfm[:, 5], VPR_MIN, VPR_MAX)
            self._coord[a] = _denorm(drfm[:, 6], 0.0, 1.0)
            switched = self._technique[a] != self._prev_technique[a]

            robot = self._robots[a]
            pos_local = robot.data.root_pos_w - self.scene.env_origins
            vel_w = robot.data.root_lin_vel_w
            self._radar_managers[a].update(
                drone_pos=pos_local,
                drone_vel=vel_w,
                technique=self._technique[a],
                pull_off_rate=self._pull_off_rate[a],
                vel_pull_off_rate=self._vel_pull_off_rate[a],
                coordination_ratio=self._coord[a],
                technique_switched=switched,
                dt=dt,
            )
            self._power[a] = (self._power[a] - self._power_cost[a][self._technique[a]]).clamp(min=0.0)

            # Update waypoint tracking
            self._waypoints[a].update(robot.data.root_pos_w)

    def _apply_action(self) -> None:
        """Write cached thrust/moment to the physics simulation for both agents."""
        if self._body_ids is None:
            self._body_ids = {a: self._robots[a].find_bodies("body")[0] for a in _AGENTS}
        for a in _AGENTS:
            self._robots[a].permanent_wrench_composer.set_forces_and_torques(
                self._thrust[a], self._moment[a], body_ids=self._body_ids[a]
            )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute 56-dim observations for each agent."""
        obs = {}
        origins = self.scene.env_origins
        for a in _AGENTS:
            robot = self._robots[a]
            wp = self._waypoints[a]
            rm = self._radar_managers[a]

            pos_w = robot.data.root_pos_w
            quat_w = robot.data.root_quat_w
            pos_local = pos_w - origins

            target_pos_b, _ = math_utils.subtract_frame_transforms(pos_w, quat_w, wp.command)
            wp_remaining = (wp.waypoints_remaining / max(_WP_PER_EP, 1)).unsqueeze(1)
            attitude = quat_w
            altitude = (pos_w[:, 2] - origins[:, 2] - _TARGET_Z).unsqueeze(1)
            vert_vel = robot.data.root_lin_vel_w[:, 2].unsqueeze(1)
            lin_vel = robot.data.root_lin_vel_b
            ang_vel = robot.data.root_ang_vel_b
            rwr = rm.get_rwr_observations(pos_local, quat_w)
            drfm_state = self._get_drfm_state_obs(a)

            obs[a] = torch.cat(
                [target_pos_b, wp_remaining, attitude, altitude, vert_vel, lin_vel, ang_vel, rwr, drfm_state],
                dim=1,
            )
        return obs

    def _get_drfm_state_obs(self, agent: str) -> torch.Tensor:
        """[N, 8]: technique one-hot(4), POR/500, VPR/200, coord, power."""
        n = self.num_envs
        obs = torch.zeros(n, 8, device=self.device)
        tech_oh = torch.zeros(n, 4, device=self.device)
        tech_oh.scatter_(1, self._technique[agent].unsqueeze(1), 1.0)
        obs[:, :4] = tech_oh
        obs[:, 4] = self._pull_off_rate[agent] / POR_MAX
        obs[:, 5] = self._vel_pull_off_rate[agent] / VPR_MAX
        obs[:, 6] = self._coord[agent]
        obs[:, 7] = self._power[agent]
        return obs

    def _get_states(self) -> torch.Tensor:
        """Global state: concatenated agent observations. Only called for custom state_space > 0."""
        return torch.cat([self.obs_dict[a] for a in _AGENTS], dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Compute per-agent rewards and return the team average for both agents."""
        r = {}
        origins = self.scene.env_origins

        for a in _AGENTS:
            robot = self._robots[a]
            wp = self._waypoints[a]
            rm = self._radar_managers[a]
            pos_w = robot.data.root_pos_w
            vel_w = robot.data.root_lin_vel_w
            quat_w = robot.data.root_quat_w

            goal = wp.command

            # -- navigation rewards
            prev_dist = torch.norm(wp._previous_pos - goal, dim=1)
            curr_dist = torch.norm(pos_w - goal, dim=1)
            prog = prev_dist - curr_dist

            target_speed = 5.0
            vec_to_goal = math_utils.normalize(goal - pos_w)
            speed_toward = (vel_w * vec_to_goal).sum(dim=1)
            fwd_speed = (speed_toward / target_speed).clamp(-1.0, 1.0)

            speed = torch.norm(vel_w, dim=1, keepdim=True).clamp(min=0.5)
            vel_dir = vel_w / speed
            heading = (vel_dir * vec_to_goal).sum(dim=1).clamp(-1.0, 1.0)

            wp_hit = wp.just_reached.float()
            comp = wp.all_done.float()

            # -- stability rewards
            up_z = (1.0 - 2.0 * (quat_w[:, 1] ** 2 + quat_w[:, 2] ** 2)).clamp(0.0, 1.0)
            height = pos_w[:, 2] - origins[:, 2]
            alt_err = torch.abs(height - _TARGET_Z)
            alt_band = (1.0 - torch.exp(-alt_err / 1.0)).clamp(0.0, 1.0)
            ang_vel_sq = torch.sum(robot.data.root_ang_vel_b ** 2, dim=1)
            act_smooth = torch.sum(self._raw_control[a] ** 2, dim=1)

            # -- safety: proximity penalty (AABB)
            drone_xy = pos_w[:, :2]
            origins_xy = origins[:, :2]
            min_surf = torch.full((self.num_envs,), 6.0, device=self.device)
            for cx, cy, hx, hy in _OBSTACLE_FOOTPRINTS.values():
                center = origins_xy + torch.tensor([cx, cy], device=self.device)
                h = torch.tensor([hx, hy], device=self.device)
                diff = torch.abs(drone_xy - center) - h
                surf = torch.norm(diff.clamp(min=0.0), dim=1)
                min_surf = torch.minimum(min_surf, surf)
            safe_dist, max_dist = 2.5, 6.0
            prox = torch.clamp((max_dist - min_surf) / (max_dist - safe_dist), 0.0, 1.0)

            # -- DRFM rewards
            max_tq = rm.track_quality.max(dim=1).values
            illumination = max_tq
            is_off = (self._technique[a] == 0).float()
            not_threatened = (max_tq < 0.3).float()
            power_con = is_off * not_threatened
            is_jamming = (self._technique[a] != 0).float()
            any_tracking = (rm.state >= 1).any(dim=1).float()
            drfm_eff = is_jamming * any_tracking * (1.0 - max_tq)
            threatened = (max_tq >= 0.3).float()
            smart_j = is_jamming * threatened

            # -- bad termination penalty (collision, too_high, radar_lock)
            crash = self._sensors[a].data.net_forces_w.norm(dim=-1).max(dim=-1).values > 0.01
            too_high_flag = (height > 5.0)
            locked = rm.any_locked
            bad_term = (crash | too_high_flag | locked).float()

            r[a] = (
                _W["progress"]         * prog
                + _W["forward_speed"]  * fwd_speed
                + _W["heading"]        * heading
                + _W["waypoint_reached"] * wp_hit
                + _W["completion_bonus"] * comp
                + _W["upright"]        * up_z
                + _W["altitude_band"]  * alt_band
                + _W["terminating"]    * bad_term
                + _W["step_penalty"]   * 1.0
                + _W["ang_vel_l2"]     * ang_vel_sq
                + _W["action_smooth"]  * act_smooth
                + _W["proximity"]      * prox
                + _W["illumination"]   * illumination
                + _W["power_conserve"] * power_con
                + _W["drfm_effective"] * drfm_eff
                + _W["smart_jam"]      * smart_j
            )

        team = torch.stack(list(r.values())).mean(dim=0)

        any_lock = torch.stack([self._radar_managers[a].any_locked for a in _AGENTS]).any(dim=0)
        max_tq_team = torch.stack(
            [self._radar_managers[a].track_quality.max(dim=1).values for a in _AGENTS]
        ).max(dim=0).values
        self.extras["metrics"] = {
            "any_lock": any_lock,
            "drfm_technique": self._technique[_AGENTS[0]],
            "illumination_penalty": max_tq_team,
        }

        return {a: team for a in _AGENTS}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute shared termination/truncation signals for both agents."""
        origins = self.scene.env_origins
        terminated = {}
        timed_out = {}

        for a in _AGENTS:
            robot = self._robots[a]
            height = robot.data.root_pos_w[:, 2] - origins[:, 2]
            crash = self._sensors[a].data.net_forces_w.norm(dim=-1).max(dim=-1).values > 0.01
            too_high_flag = height > 5.0
            locked = self._radar_managers[a].any_locked
            terminated[a] = crash | too_high_flag | locked

        any_bad = torch.stack(list(terminated.values())).any(dim=0)
        both_done = torch.stack([self._waypoints[a].all_done for a in _AGENTS]).any(dim=0)
        timeout = self.episode_length_buf >= self.max_episode_length

        terminated_shared = any_bad
        truncated_shared = timeout | both_done

        return (
            {a: terminated_shared for a in _AGENTS},
            {a: truncated_shared for a in _AGENTS},
        )

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset drones, DRFM state, radars, and waypoints for the given envs."""
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        n_reset = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        for a in _AGENTS:
            robot = self._robots[a]

            # Randomize root pose (same ranges as EventCfg.reset_base)
            root_states = robot.data.default_root_state[env_ids].clone()
            rand_pos = torch.stack([
                torch.empty(n_reset, device=self.device).uniform_(2.0, 6.0),   # x
                torch.empty(n_reset, device=self.device).uniform_(-10.0, 10.0),  # y
                torch.zeros(n_reset, device=self.device),                        # z (additive to default 0.5)
            ], dim=1)
            rand_euler = torch.stack([
                torch.empty(n_reset, device=self.device).uniform_(-0.1, 0.1),   # roll
                torch.empty(n_reset, device=self.device).uniform_(-0.1, 0.1),   # pitch
                torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2),   # yaw
            ], dim=1)
            positions = root_states[:, 0:3] + origins + rand_pos
            orient_delta = math_utils.quat_from_euler_xyz(rand_euler[:, 0], rand_euler[:, 1], rand_euler[:, 2])
            orientations = math_utils.quat_mul(root_states[:, 3:7], orient_delta)
            velocities = torch.zeros(n_reset, 6, device=self.device)

            robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
            robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)

            # Reset joint state
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            robot.reset(env_ids)

            # Reset motor, DRFM tensors
            self._motor[a].reset(env_ids)
            self._technique[a][env_ids] = 0
            self._prev_technique[a][env_ids] = 0
            self._pull_off_rate[a][env_ids] = 0.0
            self._vel_pull_off_rate[a][env_ids] = 0.0
            self._coord[a][env_ids] = 0.0
            self._power[a][env_ids] = 1.0
            self._raw_control[a][env_ids] = 0.0
            self._thrust[a][env_ids] = 0.0
            self._moment[a][env_ids] = 0.0

            # Reset contact sensor
            self._sensors[a].data.net_forces_w[env_ids] = 0.0

            # Reset radar manager
            self._radar_managers[a].reset(env_ids)

            # Reset waypoint tracker (use reset positions as previous_pos)
            pos_w_approx = positions  # close enough; actual pos set above
            # Build a full-size tensor so WaypointTracker can index it
            dummy_pos = self._robots[a].data.root_pos_w.clone()
            dummy_pos[env_ids] = positions
            self._waypoints[a].reset(env_ids, dummy_pos, self.scene.env_origins)
