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
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Linear velocity of the drone in its body frame, shape [N, 3]."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocity of the drone in its body frame, shape [N, 3]."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def root_quat_w(
    env: ManagerBasedRLEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Attitude quaternion (w, x, y, z) in world frame, shape [N, 4]; optionally hemispherically unique."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def target_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Current waypoint position expressed in the drone body frame, shape [N, 3]."""
    asset: Articulation = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos_tensor = env.command_manager.get_term(command_name).command[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    pos_b, _ = math_utils.subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor)
    return pos_b


def waypoints_remaining(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Fraction of waypoints still to visit in [0, 1], shape [N, 1]; 0.0 when all are done."""
    cmd = env.command_manager.get_term(command_name)
    remaining = cmd.waypoints_remaining.unsqueeze(-1)
    total = cmd.cfg.waypoints_per_episode
    return remaining / max(total, 1)


def rwr_observations(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Realistic RWR observations for all three radars, shape [N, 32]."""
    drfm = env.action_manager.get_term("drfm_action")
    robot = env.scene["robot"]
    return drfm.radar_manager.get_rwr_observations(
        robot.data.root_pos_w - env.scene.env_origins,
        robot.data.root_quat_w,
    )


def altitude_obs(
    env: ManagerBasedRLEnv,
    target_z: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Height above ground relative to target altitude, shape [N, 1]."""
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height - target_z).unsqueeze(-1)


def vertical_vel_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """World-frame vertical velocity, shape [N, 1]."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 2].unsqueeze(-1)


def drfm_state_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """DRFM technique one-hot, normalised POR/VPOR, coordination ratio, and power, shape [N, 8]."""
    return env.action_manager.get_term("drfm_action").get_state_obs()
