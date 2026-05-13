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
    obstacle_names: list,
    safe_dist: float = 2.5,
    max_dist: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty in [0, 1] that rises as drone enters the safe_dist bubble around any obstacle."""
    asset: Articulation = env.scene[asset_cfg.name]
    drone_pos = asset.data.root_pos_w

    min_dist = torch.full((env.num_envs,), max_dist, device=env.device)
    for name in obstacle_names:
        obstacle: RigidObject = env.scene[name]
        obs_pos = obstacle.data.root_pos_w
        dist = torch.norm(drone_pos - obs_pos, dim=1)
        min_dist = torch.minimum(min_dist, dist)

    return torch.clamp((max_dist - min_dist) / (max_dist - safe_dist), 0.0, 1.0)


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
    speed = torch.norm(drone_vel, dim=1, keepdim=True).clamp(min=1e-6)
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


def distance_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Euclidean distance to current waypoint in metres; used as a penalty with a negative weight."""
    asset: Articulation = env.scene[asset_cfg.name]
    goal = env.command_manager.get_term(command_name).command[:, :3]
    dist = torch.norm(asset.data.root_pos_w - goal, dim=1)
    return dist


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
