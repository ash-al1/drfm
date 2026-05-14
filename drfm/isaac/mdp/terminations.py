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


def too_low(
    env: ManagerBasedRLEnv,
    min_z: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when drone drops below min_z metres above its environment origin."""
    asset: RigidObject = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return height < min_z


def all_waypoints_done(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """True when every waypoint in the episode has been visited."""
    return env.command_manager.get_term(command_name).all_done

def radar_lock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when any radar has achieved a firm lock on the drone (tq threshold exceeded)."""
    return env.action_manager.get_term("drfm_action").radar_manager.any_locked


_BAD_TERM_NAMES = ("collision", "too_high", "radar_lock")


def is_bad_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True only when the episode ends due to collision, too_high, or radar_lock - not timeout or success."""
    tm = env.termination_manager
    active = [name for name in _BAD_TERM_NAMES if name in tm._term_name_to_term_idx]
    if not active:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.stack([tm.get_term(name) for name in active], dim=1).any(dim=1)
