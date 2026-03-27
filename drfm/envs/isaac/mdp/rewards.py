#

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: object = env.scene[asset_cfg.name]

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
    asset: object = env.scene[asset_cfg.name]
    drone_pos = asset.data.root_pos_w

    min_dist = torch.full((env.num_envs,), max_dist, device=env.device)
    for name in obstacle_names:
        obs_pos = env.scene[name].data.root_pos_w
        dist = torch.norm(drone_pos - obs_pos, dim=1)
        min_dist = torch.minimum(min_dist, dist)

    return torch.clamp((max_dist - min_dist) / (max_dist - safe_dist), 0.0, 1.0)


def heading_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: object = env.scene[asset_cfg.name]
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
    asset: object = env.scene[asset_cfg.name]
    goal = env.command_manager.get_term(command_name).command[:, :3]
    dist = torch.norm(asset.data.root_pos_w - goal, dim=1)
    return (dist < threshold).float()


def completion_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    return cmd.all_done.float()


def step_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def distance_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: object = env.scene[asset_cfg.name]
    goal = env.command_manager.get_term(command_name).command[:, :3]
    dist = torch.norm(asset.data.root_pos_w - goal, dim=1)
    return dist


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: object = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
