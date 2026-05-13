# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali
#
# Copyright (c) 2025, Kousheek Chakraborty
# Original work licensed under the BSD-3-Clause License.
# Built on the IsaacLab framework (https://github.com/isaac-sim/IsaacLab).

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reduction in distance to goal this step; positive when drone moves toward goal."""
    asset: Articulation = env.scene[asset_cfg.name]

    target_pos = env.command_manager.get_term(command_name).command[:, :3]
    previous_pos = env.command_manager.get_term(command_name).previous_pos
    current_pos = asset.data.root_pos_w

    prev_distance = torch.norm(previous_pos - target_pos, dim=1)
    current_distance = torch.norm(current_pos - target_pos, dim=1)

    return prev_distance - current_distance


def proximity_penalty(
    env: ManagerBasedRLEnv,
    obstacle_footprints: dict[str, tuple[float, float, float, float]],
    safe_dist: float = 2.5,
    max_dist: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty in [0, 1] based on AABB surface distance to the nearest obstacle.

    obstacle_footprints maps obstacle name -> (cx, cy, half_x, half_y) in env-local coords.
    Returns 1.0 when on the surface, 0.0 at max_dist or beyond.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    drone_xy = asset.data.root_pos_w[:, :2]
    origins_xy = env.scene.env_origins[:, :2]

    min_surf_dist = torch.full((env.num_envs,), max_dist, device=env.device)
    for cx, cy, hx, hy in obstacle_footprints.values():
        center = origins_xy + torch.tensor([cx, cy], device=env.device)
        h = torch.tensor([hx, hy], device=env.device)
        diff = torch.abs(drone_xy - center) - h
        surf_dist = torch.norm(diff.clamp(min=0.0), dim=1)
        min_surf_dist = torch.minimum(min_surf_dist, surf_dist)

    return torch.clamp((max_dist - min_surf_dist) / (max_dist - safe_dist), 0.0, 1.0)


def heading_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Alignment of velocity with goal direction in [0, 1]; 1.0 means flying directly toward goal."""
    asset: Articulation = env.scene[asset_cfg.name]
    drone_pos = asset.data.root_pos_w
    drone_vel = asset.data.root_lin_vel_w
    goal = env.command_manager.get_term(command_name).command[:, :3]

    vec_to_goal = math_utils.normalize(goal - drone_pos)
    speed = torch.norm(drone_vel, dim=1, keepdim=True).clamp(min=0.5)
    vel_dir = drone_vel / speed

    dot = (vel_dir * vec_to_goal).sum(dim=1).clamp(-1.0, 1.0)
    return (dot + 1.0) * 0.5


def arrived(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """1.0 when drone is within threshold metres of the current waypoint, else 0.0."""
    asset: Articulation = env.scene[asset_cfg.name]
    goal = env.command_manager.get_term(command_name).command[:, :3]
    dist = torch.norm(asset.data.root_pos_w - goal, dim=1)
    return (dist < threshold).float()


def completion_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """1.0 on the step all waypoints are completed, else 0.0."""
    cmd = env.command_manager.get_term(command_name)
    return cmd.all_done.float()


def step_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 per step; multiplied by a negative weight in the config to discourage idling."""
    return torch.ones(env.num_envs, device=env.device)


def forward_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_speed: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Speed toward goal normalised by target_speed, clamped to [0, 1]; positive when closing on goal."""
    asset: Articulation = env.scene[asset_cfg.name]
    goal  = env.command_manager.get_term(command_name).command[:, :3]
    vec_to_goal = math_utils.normalize(goal - asset.data.root_pos_w)
    speed_toward = (asset.data.root_lin_vel_w * vec_to_goal).sum(dim=1)
    return (speed_toward / target_speed).clamp(0.0, 1.0)


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared L2 norm of body-frame angular velocity; used as a penalty to discourage spinning."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)


def illumination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Continuous penalty proportional to max track_quality across radars; proxy for illumination threat."""
    drfm = env.action_manager.get_term("drfm_action")
    return drfm.radar_manager.track_quality.max(dim=1).values


def power_conserve(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for choosing technique=OFF when no radar has track_quality >= 0.3."""
    drfm = env.action_manager.get_term("drfm_action")
    is_off = (drfm._technique == 0).float()
    not_threatened = (drfm.radar_manager.track_quality.max(dim=1).values < 0.3).float()
    return is_off * not_threatened


def upright_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for keeping roll and pitch small. Returns 1.0 when perfectly level, 0.0 at 90 deg tilt."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    up_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    return up_z.clamp(0.0, 1.0)


def altitude_hold(
    env: ManagerBasedRLEnv,
    target_z: float = 2.0,
    tolerance: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for staying near target altitude"""
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    error = torch.abs(height - target_z)
    return torch.exp(-error / tolerance)


def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared L2 norm of raw actions; penalizes large control inputs."""
    actions = env.action_manager.get_term("control_action").raw_actions
    return torch.sum(torch.square(actions), dim=1)


def waypoint_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Fractional reward based on waypoints visited so far."""
    cmd = env.command_manager.get_term(command_name)
    total = cmd.cfg.waypoints_per_episode
    visited = cmd._waypoints_visited.float()
    return visited / max(total, 1)


def drfm_effectiveness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for actively reducing track_quality."""
    drfm = env.action_manager.get_term("drfm_action")
    rm = drfm.radar_manager
    is_jamming = (drfm._technique != 0).float()
    max_tq = rm.track_quality.max(dim=1).values
    any_tracking = (rm.state >= 1).any(dim=1).float()
    return is_jamming * any_tracking * (1.0 - max_tq)


def smart_jamming(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for activating DRFM when actually threatened (track_quality >= 0.3)"""
    drfm = env.action_manager.get_term("drfm_action")
    rm = drfm.radar_manager
    is_jamming = (drfm._technique != 0).float()
    threatened = (rm.track_quality.max(dim=1).values >= 0.3).float()
    return is_jamming * threatened
