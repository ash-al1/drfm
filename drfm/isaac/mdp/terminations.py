# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali
#
# Copyright (c) 2025, Kousheek Chakraborty
# Original work licensed under the BSD-3-Clause License.
# Built on the IsaacLab framework (https://github.com/isaac-sim/IsaacLab).

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def too_high(
    env: ManagerBasedRLEnv,
    max_z: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when drone exceeds max_z metres above its environment origin."""
    asset: RigidObject = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return height > max_z


def all_waypoints_done(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """True when every waypoint in the episode has been visited."""
    return env.command_manager.get_term(command_name).all_done


def flyaway(
    env: ManagerBasedRLEnv,
    distance: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when drone is more than distance metres from the reference point (waypoint or fixed pos)."""
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos_tensor = env.command_manager.get_term(command_name).command[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    distance_tensor = torch.linalg.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return distance_tensor > distance


def radar_lock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when any radar has achieved a firm lock on the drone (tq threshold exceeded)."""
    return env.action_manager.get_term("drfm_action").radar_manager.any_locked
